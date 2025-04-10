// TTSManager.swift
// Local STT Model Runner
// Created by Dari Gomez on 4/10/25.

import Foundation
import SwiftUI

class TTSManager {
    // Holds a reference to the Whisper wrapper.
    var whisper: WhisperWrapper?
    
    init() {
        // Any additional setup can go here.
    }
    
    /// Begins transcription of the audio file at filePath.
    /// - Parameters:
    ///   - filePath: The local URL string to the recorded audio file.
    ///   - callback: A closure that receives the transcription result.
    func beginTranscription(_ filePath: String, callback: @escaping (String) -> Void) {
        // Locate the model file in the bundle.
        guard let modelPath = Bundle.main.path(forResource: "ggml-small.en", ofType: "bin") else {
            callback("Model file not found in bundle.")
            return
        }
        
        // Debug print: show the model path.
        print("Model path: \(modelPath)")
        
        // Initialize the Whisper wrapper if it's not already initialized.
        if whisper == nil {
            whisper = WhisperWrapper(modelPath: modelPath)
        }
        
        // Run transcription on a background thread.
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self,
                  let transcript = self.whisper?.transcribeAudio(atPath: filePath) else {
                DispatchQueue.main.async {
                    callback("Transcription failed.")
                }
                return
            }
            DispatchQueue.main.async {
                callback(transcript)
            }
        }
    }
}
