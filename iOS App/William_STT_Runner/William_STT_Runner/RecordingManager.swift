//
//  RecordingManager.swift
//  William_STT_Runner
//
//  Created by Dari Gomez on 4/8/25.
//

import Foundation
import AVFAudio
import SwiftUI
import Torch  // Import Torch (from LibTorch-Lite)

class RecordingManager: ObservableObject {
    var audioRecorder: AVAudioRecorder?
    var microphoneAllowed: Bool = false
    var torchModule: TorchModule?  // TorchScript model loaded here
    
    // MARK: - Initialization
    
    init(status: Status) {
        self.status = status
        setStatus(status)
        transcription = "Checking microphone availability..."
        requestMicrophoneAccess()
        loadTorchModel()  // Load the TorchScript model upon initialization
    }
    
    // MARK: - Torch Model Loading
    
    func loadTorchModel() {
        guard let modelPath = Bundle.main.path(forResource: "finetuned_whisper_traced", ofType: "pt") else {
            print("Error: TorchScript model file not found in bundle!")
            return
        }
        do {
            torchModule = try TorchModule(fileAtPath: modelPath)
            print("TorchScript model loaded successfully.")
        } catch {
            print("Error loading TorchScript model: \(error)")
        }
    }
    
    // MARK: - Microphone and Recording Methods
    
    func requestMicrophoneAccess() {
        AVAudioApplication.requestRecordPermission { [weak self] allowed in
            DispatchQueue.main.async {
                if allowed {
                    self?.microphoneAllowed = true
                    self?.setStatus(.ready)
                } else {
                    print("Microphone access denied")
                    self?.setStatus(.unavailable)
                    self?.showMicrophonePermissionAlert = true
                }
            }
        }
    }
    
    func startRecording() {
        if !microphoneAllowed { return }
        
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default)
            try session.setActive(true)
            
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 44100.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsFloatKey: false
            ]
            
            guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                print("❌ Could not find the documents directory")
                return
            }
            
            let fileName = "willAudio_YYYY-MM-DD_start-HHMMSS_duration-HHMMSS_appTest.wav"
            let filePath = documentsURL.appendingPathComponent(fileName)
            
            audioRecorder = try AVAudioRecorder(url: filePath, settings: settings)
            audioRecorder?.prepareToRecord()
            audioRecorder?.record()
            
            print("Recording started, saving to \(filePath.path)")
            
            // Stop recording after 10 seconds (for testing)
            DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
                self.stopRecording()
            }
            
        } catch {
            print("Failed to set up recorder:", error)
        }
    }
    
    func stopRecording() {
        audioRecorder?.stop()
        print("Recording stopped")
        // After stopping, attempt to transcribe the recording.
        // (In a complete implementation, you'll preprocess the audio file into a mel spectrogram.)
        if let fileURL = audioRecorder?.url {
            transcribeRecording(fileURL: fileURL)
        }
    }
    
    // MARK: - TorchScript Inference (Transcription)
    
    /// This function simulates running inference on the recorded audio.
    /// Replace the dummy tensor with your actual preprocessed input in the future.
    func transcribeRecording(fileURL: URL) {
        // Placeholder: Here you would load and preprocess the audio file at fileURL.
        // For now, we create a dummy input tensor representing a mel spectrogram of shape [1, 80, 3000].
        let dummyData = [Float](repeating: 0.0, count: 1 * 80 * 3000)
        let dummyInput = Tensor(data: dummyData, shape: [1, 80, 3000])
        
        guard let module = torchModule else {
            print("Torch model not loaded.")
            DispatchQueue.main.async {
                self.transcription = "Error: Model not loaded."
            }
            return
        }
        
        do {
            // Run inference with the dummy input.
            let output = try module.predict(withInputs: [dummyInput])
            // Placeholder: Replace this with proper decoding of the model's output into text.
            let outputString = "Simulated transcription: \(output)"
            
            DispatchQueue.main.async {
                self.transcription = outputString
                self.setStatus(.ready)
            }
        } catch {
            print("Error during inference:", error)
            DispatchQueue.main.async {
                self.transcription = "Error during transcription: \(error)"
            }
        }
    }
    
    // MARK: - Status Management
    
    enum Status {
        case ready
        case active
        case unavailable
        case busy
    }
    
    var status: Status
    
    fileprivate func setStatus(_ status: Status) {
        switch status {
        case .ready:
            buttonColor = .blue
            buttonIcon = "microphone.fill"
            buttonLabel = "Record"
            buttonDisabled = false
            transcription = "Press Record to begin."
        case .active:
            buttonColor = .red
            buttonIcon = "stop.fill"
            buttonLabel = "Stop"
            buttonDisabled = false
            transcription = "Recording in progress..."
        case .unavailable:
            buttonColor = .gray
            buttonIcon = "microphone.slash.fill"
            buttonLabel = "Unavailable"
            buttonDisabled = true
            transcription = "The application does not have microphone access. Please grant it in settings."
        case .busy:
            buttonColor = .gray
            buttonIcon = "clock.arrow.trianglehead.2.counterclockwise.rotate.90"
            buttonLabel = "Please Wait"
            buttonDisabled = true
            transcription = "Transcribing..."
        }
        self.status = status
    }
    
    func OnPressButton() {
        switch status {
        case .ready:
            setStatus(.active)
            startRecording()
        case .active:
            setStatus(.ready)
            stopRecording()
        default:
            fatalError("Unknown status. Is the button disabled?")
        }
    }
    
    // MARK: - Published Properties
    
    @Published var buttonColor: Color = .gray
    @Published var buttonIcon: String = ""
    @Published var buttonLabel: String = "Please Wait"
    @Published var buttonDisabled = false
    @Published var showMicrophonePermissionAlert = false
    @Published var transcription: String = "Transcription Unavailable"
}
