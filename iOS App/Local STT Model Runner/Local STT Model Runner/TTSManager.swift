//
//  TTSManager.swift
//  Local STT Model Runner
//
//  Created by Dari Gomez on 4/10/25.
//

import Foundation
import Speech

class TTSManager {
    // Initialize a speech recognizer using the current locale (adjust the locale identifier if needed).
    private let speechRecognizer: SFSpeechRecognizer?
    
    init() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    }
    
    // Updated beginTranscription function now uses Apple's Speech framework.
    func beginTranscription(_ filePath: String, callback: @escaping (String) -> Void) {
        // Attempt to create a URL from the file path.
        guard let audioURL = URL(string: filePath) else {
            callback("Invalid file path.")
            return
        }
        
        // Create a recognition request from the audio file URL.
        let recognitionRequest = SFSpeechURLRecognitionRequest(url: audioURL)
        
        // Ensure the speech recognizer is available.
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            callback("Speech recognizer is not available.")
            return
        }
        
        // Request speech recognition authorization.
        SFSpeechRecognizer.requestAuthorization { authStatus in
            // Return to the main thread for UI updates.
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    // Start the recognition task.
                    recognizer.recognitionTask(with: recognitionRequest) { result, error in
                        if let error = error {
                            print("Recognition error: \(error.localizedDescription)")
                            callback("Error transcribing audio: \(error.localizedDescription)")
                        } else if let result = result, result.isFinal {
                            let transcription = result.bestTranscription.formattedString
                            print("Transcription: \(transcription)")
                            callback(transcription)
                        }
                    }
                case .denied:
                    callback("Speech recognition authorization denied.")
                case .restricted:
                    callback("Speech recognition restricted on this device.")
                case .notDetermined:
                    callback("Speech recognition not yet authorized.")
                @unknown default:
                    callback("Speech recognition authorization status unknown.")
                }
            }
        }
    }
}
