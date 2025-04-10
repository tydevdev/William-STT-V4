//
//  RecordingManager.swift
//  Local STT Model Runner
//
//  Created by Dari Gomez on 4/8/25.
//

import Foundation
import AVFAudio
import SwiftUI

class RecordingManager : ObservableObject {
    var audioRecorder: AVAudioRecorder?
    var microphoneAllowed : Bool = false
    var tts : TTSManager
    
    init(status: Status) {
        self.status = status
        tts = TTSManager()
        setStatus(status)
        transcription = "Checking microphone availability..."
        requestMicrophoneAccess()
    }
    
    func requestMicrophoneAccess() {
        AVAudioApplication.requestRecordPermission { [weak self] allowed in
            DispatchQueue.main.async {
                if allowed {
                    self?.microphoneAllowed = true;
                    self?.setStatus(Status.ready)
                } else {
                    print("Microphone access denied")
                    self?.setStatus(Status.unavailable)
                    self?.showMicrophonePermissionAlert = true
                }
            }
        }
    }
    
    private var recordingPath : URL?

    func startRecording() {
        if !microphoneAllowed {
            return
        }
        
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
                    print("Could not find the documents directory")
                    return
                }
            
            let startTime = Date.now

            let fileName = "willAudio_\(startTime.ISO8601Format())_appTest.wav"
            let filePath = documentsURL.appendingPathComponent(fileName)
            recordingPath = filePath

            audioRecorder = try AVAudioRecorder(url: filePath, settings: settings)
            audioRecorder?.prepareToRecord()
            audioRecorder?.record()

            print("Recording started, saving to \(filePath.path)")

        } catch {
            print("Failed to set up recorder:", error)
        }
    }

    func stopRecording() {
        audioRecorder?.stop()
        print("Recording stopped")
        tts.beginTranscription(recordingPath!.absoluteString, callback: showTranscription)
    }
    
    func showTranscription(_ transcription : String) {
        print(transcription)
        setStatus(Status.ready)
        self.transcription = transcription
    }
    
    enum Status {
        case ready
        case active
        case unavailable
        case busy
    }
    
    var status : Status
    
    fileprivate func setStatus(_ status: Status) {
        switch status {
        case Status.ready:
            buttonColor = .blue
            buttonIcon = "microphone.fill"
            buttonLabel = "Record"
            buttonDisabled = false
            transcription = "Press Record to begin."
        case Status.active:
            buttonColor = .red
            buttonIcon = "stop.fill"
            buttonLabel = "Stop"
            buttonDisabled = false
            transcription = "Recording in progress..."
        case Status.unavailable:
            buttonColor = .gray
            buttonIcon = "microphone.slash.fill"
            buttonLabel = "Unavailable"
            buttonDisabled = true
            transcription = "The application does not have microphone access. Please grant it in settings."
        case Status.busy:
            buttonColor = .gray
            buttonIcon = "clock.arrow.trianglehead.2.counterclockwise.rotate.90"
            buttonLabel = "Please Wait"
            buttonDisabled = true
            transcription = "Transcribing..."
        }
        self.status = status;
    }
    
    func OnPressButton() {
        switch status {
        case Status.ready:
            setStatus(Status.active)
            startRecording()
        case Status.active:
            setStatus(Status.busy)
            stopRecording()
        default:
            fatalError("Unkown status. Is the button disabled?")
        }
    }
    
    @Published var buttonColor : Color = .gray
    @Published var buttonIcon : String = ""
    @Published var buttonLabel : String = "Please Wait"
    @Published var buttonDisabled = false
    @Published var showMicrophonePermissionAlert = false
    
    @Published var transcription : String = "Transcription Unavailable";
}
