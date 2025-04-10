//
//  WhisperWrapper.h
//  Local STT Model Runner
//
//  Created by Ty DeVito on 4/10/25.
//


//
//  WhisperWrapper.h
//  Local STT Model Runner
//
//  A simple wrapper to expose Whisper functionality to Swift.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface WhisperWrapper : NSObject

// Initializes the Whisper model with the specified model file path.
- (instancetype)initWithModelPath:(NSString *)modelPath;

// Transcribe an audio file at the given path.
// Returns the transcription as an NSString.
- (NSString *)transcribeAudioAtPath:(NSString *)audioPath;

@end

NS_ASSUME_NONNULL_END
