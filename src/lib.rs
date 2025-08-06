//! Head-related transfer function (HRTF) node.
//!
//! This crate utilizes Fyrox's [hrtf](https://docs.rs/hrtf/latest/hrtf/)
//! crate, with impulse response data from the
//! [IRCAM database](http://recherche.ircam.fr/equipes/salles/listen/download.html).

use firewheel::{
    channel_config::{ChannelConfig, NonZeroChannelCount},
    diff::{Diff, Patch},
    log::RealtimeLogger,
    node::{AudioNode, AudioNodeInfo, AudioNodeProcessor, ProcBuffers, ProcessStatus},
};
use glam::Vec3;
use hrtf::{HrirSphere, HrtfContext, HrtfProcessor};
use std::io::Cursor;

mod subjects;

pub use subjects::{Subject, SubjectBytes};

/// Head-related transfer function (HRTF) node.
#[derive(Debug, Default, Clone, Diff, Patch)]
#[cfg_attr(feature = "bevy", derive(bevy_ecs::component::Component))]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub struct FyroxHrtfNode {
    /// The positional offset from the listener to the emitter.
    pub offset: Vec3,
}

/// Configuration for [`FyroxHrtfNode`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy_ecs::component::Component))]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub struct HrtfConfig {
    /// The number of input channels.
    ///
    /// The inputs are downmixed to a mono signal
    /// before spatialization is applied.
    ///
    /// Defaults to [`NonZeroChannelCount::STEREO`].
    pub input_channels: NonZeroChannelCount,

    /// The head-related impulse-response sphere.
    ///
    /// The data for this sphere is captured from subjects. Short
    /// "impulses" are played from all angles and recorded at the
    /// ear canal. The resulting recordings capture how sounds are affected
    /// by the subject's torso, head, and ears.
    ///
    /// The effect can be rather personal as a result, but the
    /// `IRC_1040` subject is commonly sited as a good default.
    pub hrir_sphere: HrirSource,

    /// The size of the FFT processing block, which can be
    /// tuned for performance.
    pub fft_size: FftSize,
}

impl Default for HrtfConfig {
    fn default() -> Self {
        Self {
            input_channels: NonZeroChannelCount::STEREO,
            hrir_sphere: Subject::Irc1040.into(),
            fft_size: FftSize::default(),
        }
    }
}

/// Describes the size of the FFT processing block.
///
/// Generally, you should try to match the FFT size (the product of
/// [`slice_count`][FftSize::slice_count] and [`slice_len`][FftSize::slice_len])
/// to the audio's processing buffer size if possible.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub struct FftSize {
    /// The number of slices the audio stream is split into for overlap-save.
    ///
    /// Defaults to 4.
    slice_count: usize,

    /// The size of each slice.
    ///
    /// Defaults to 128.
    slice_len: usize,
}

impl Default for FftSize {
    fn default() -> Self {
        Self {
            slice_count: 4,
            slice_len: 128,
        }
    }
}

/// Provides a source for the HRIR sphere data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum HrirSource {
    /// Load data from the subjects embedded in the binary itself.
    Embedded(Subject),
    /// Load arbitrary data from an in-memory slice.
    InMemory(SubjectBytes),
}

impl HrirSource {
    fn get_sphere(&self, sample_rate: u32) -> Result<HrirSphere, hrtf::HrtfError> {
        match &self {
            HrirSource::Embedded(subject) => HrirSphere::new(Cursor::new(*subject), sample_rate),
            HrirSource::InMemory(subject) => {
                HrirSphere::new(Cursor::new(subject.clone()), sample_rate)
            }
        }
    }
}

impl From<Subject> for HrirSource {
    fn from(value: Subject) -> Self {
        Self::Embedded(value)
    }
}

impl From<SubjectBytes> for HrirSource {
    fn from(value: SubjectBytes) -> Self {
        Self::InMemory(value)
    }
}

impl AudioNode for FyroxHrtfNode {
    type Configuration = HrtfConfig;

    fn info(&self, config: &Self::Configuration) -> AudioNodeInfo {
        AudioNodeInfo::new()
            .debug_name("hrtf node")
            .channel_config(ChannelConfig::new(config.input_channels.get(), 2))
    }

    fn construct_processor(
        &self,
        config: &Self::Configuration,
        cx: firewheel::node::ConstructProcessorContext,
    ) -> impl firewheel::node::AudioNodeProcessor {
        let sample_rate = cx.stream_info.sample_rate.get();

        let sphere = config
            .hrir_sphere
            .get_sphere(sample_rate)
            .expect("HRIR data should be in a valid format");

        let fft_buffer_len = config.fft_size.slice_count * config.fft_size.slice_len;

        let renderer = HrtfProcessor::new(
            sphere,
            config.fft_size.slice_count,
            config.fft_size.slice_len,
        );

        let buffer_size = cx.stream_info.max_block_frames.get() as usize;
        FyroxHrtfProcessor {
            renderer,
            offset: self.offset,
            distance: 1.0,
            fft_input: Vec::with_capacity(fft_buffer_len),
            fft_output: Vec::with_capacity(buffer_size.max(fft_buffer_len)),
            prev_left_samples: Vec::with_capacity(fft_buffer_len),
            prev_right_samples: Vec::with_capacity(fft_buffer_len),
            sphere_source: config.hrir_sphere.clone(),
            fft_size: config.fft_size.clone(),
        }
    }
}

struct FyroxHrtfProcessor {
    renderer: HrtfProcessor,
    offset: Vec3,
    distance: f32,
    fft_input: Vec<f32>,
    fft_output: Vec<(f32, f32)>,
    prev_left_samples: Vec<f32>,
    prev_right_samples: Vec<f32>,
    sphere_source: HrirSource,
    fft_size: FftSize,
}

/// Here we utilize the same fall-off that Firewheel's
/// built-in `SpatialBasic` node does.
fn distance_gain(distance: f32) -> f32 {
    10.0f32.powf(-0.03 * distance)
}

impl AudioNodeProcessor for FyroxHrtfProcessor {
    fn process(
        &mut self,
        ProcBuffers {
            inputs, outputs, ..
        }: ProcBuffers,
        proc_info: &firewheel::node::ProcInfo,
        events: &mut firewheel::event::NodeEventList,
        _: &mut RealtimeLogger,
    ) -> ProcessStatus {
        let mut previous_vector = self.offset;
        let mut previous_gain = distance_gain(self.distance);

        for FyroxHrtfNodePatch::Offset(offset) in events.drain_patches::<FyroxHrtfNode>() {
            // TODO: this might not be correct if the event rate is faster
            // than the FFT processing rate.
            self.distance = offset.length().max(0.01);
            self.offset = offset.normalize_or(Vec3::Y);
        }

        if proc_info.in_silence_mask.all_channels_silent(inputs.len()) {
            return ProcessStatus::ClearAllOutputs;
        }

        let current_gain = distance_gain(self.distance);
        for frame in 0..proc_info.frames {
            let mut downmixed = 0.0;
            for channel in inputs {
                downmixed += channel[frame];
            }
            downmixed /= inputs.len() as f32;

            self.fft_input.push(downmixed);

            // Buffer full, process FFT
            if self.fft_input.len() == self.fft_input.capacity() {
                let fft_len = self.fft_input.len();

                let output_start = self.fft_output.len();
                self.fft_output
                    .extend(std::iter::repeat_n((0.0, 0.0), fft_len));

                // let (left, right) = outputs.split_at_mut(1);
                let context = HrtfContext {
                    source: &self.fft_input,
                    output: &mut self.fft_output[output_start..],
                    new_sample_vector: hrtf::Vec3::new(self.offset.x, self.offset.y, self.offset.z),
                    prev_sample_vector: hrtf::Vec3::new(
                        previous_vector.x,
                        previous_vector.y,
                        previous_vector.z,
                    ),
                    prev_left_samples: &mut self.prev_left_samples,
                    prev_right_samples: &mut self.prev_right_samples,
                    new_distance_gain: current_gain,
                    prev_distance_gain: previous_gain,
                };

                self.renderer.process_samples(context);

                // in case we call this multiple times
                previous_vector = self.offset;
                previous_gain = current_gain;
                self.fft_input.clear();
            }
        }

        for (i, (left, right)) in self
            .fft_output
            .drain(..proc_info.frames.min(self.fft_output.len()))
            .enumerate()
        {
            outputs[0][i] = left;
            outputs[1][i] = right;
        }

        ProcessStatus::outputs_not_silent()
    }

    fn new_stream(&mut self, stream_info: &firewheel::StreamInfo) {
        if stream_info.prev_sample_rate != stream_info.sample_rate {
            let sample_rate = stream_info.sample_rate.get();

            let sphere = self
                .sphere_source
                .get_sphere(sample_rate)
                .expect("HRIR data should be in a valid format");

            let renderer =
                HrtfProcessor::new(sphere, self.fft_size.slice_count, self.fft_size.slice_len);

            self.renderer = renderer;
        }
    }
}
