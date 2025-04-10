//
//  WhisperWrapper.mm
//  Local STT Model Runner
//
//  A simple Objective-C++ wrapper for the Whisper.cpp API.
//  This code initializes the model, runs inference, and retrieves the transcription
//  as a concatenated string of all output segments.
//
//  IMPORTANT: Implement actual audio loading to fill `audioData` with PCM float samples.
//
 
#import "WhisperWrapper.h"
#include "whisper.h"  // Ensure your include paths are set correctly for Whisper headers.
#include <string>
#include <vector>
 
@interface WhisperWrapper () {
    struct whisper_context *ctx;
}
@end
 
@implementation WhisperWrapper
 
- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        // Convert NSString modelPath to std::string.
        std::string model_path([modelPath UTF8String]);
        // Initialize the Whisper model from the file.
        ctx = whisper_init_from_file(model_path.c_str());
        if (ctx == nullptr) {
            NSLog(@"[WhisperWrapper] Failed to initialize the Whisper model at: %s", [modelPath UTF8String]);
            return nil;
        }
    }
    return self;
}
 
- (NSString *)transcribeAudioAtPath:(NSString *)audioPath {
    // --- Load Audio ---
    // Replace this placeholder with proper audio loading code.
    // For example, use AVFoundation to read the WAV file at 'audioPath'
    // and convert it into 16kHz 32-bit float PCM stored in audioData.
    std::vector<float> audioData;
    // TODO: Implement a proper audio file loader.
    
    // --- Set Up Inference Parameters ---
    // Use default parameters for greedy sampling.
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = "en"; // Set language to English (if using an English-only model)
    
    // --- Run Inference ---
    // Call whisper_full with parameters passed by value.
    int ret = whisper_full(ctx, params, audioData.data(), (int)audioData.size());
    if (ret != 0) {
        NSLog(@"[WhisperWrapper] whisper_full failed with code %d", ret);
        return @"";
    }
    
    // --- Collect and Concatenate Transcript Segments ---
    int n_segments = whisper_full_n_segments(ctx);
    std::string finalTranscript;
    for (int i = 0; i < n_segments; i++) {
        // Retrieve text for each segment.
        const char *seg_text = whisper_full_get_segment_text(ctx, i);
        if (seg_text) {
            finalTranscript += seg_text;
            finalTranscript += " "; // Add a space between segments.
        }
    }
    
    return [NSString stringWithUTF8String:finalTranscript.c_str()];
}
 
- (void)dealloc {
    if (ctx != nullptr) {
        whisper_free(ctx);
        ctx = nullptr;
    }
}
 
@end
