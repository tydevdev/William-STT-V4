//
//  RecordingManager.swift
//  Local STT Model Runner
//
//  Created by Dari Gomez on 4/8/25.
//

import Foundation
import AVFAudio

class RecordingManager {
    var recorder : AVAudioRecorder
    
    init(recorder: AVAudioRecorder) {
        self.recorder = recorder
    }
}
