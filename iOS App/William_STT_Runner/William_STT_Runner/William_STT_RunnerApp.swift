//
//  William_STT_RunnerApp.swift
//  William_STT_Runner
//
//  Created by Ty DeVito on 3/31/25.
//

import SwiftUI


@main
struct William_STT_RunnerApp: App {
    @StateObject var recordingManager = RecordingManager(status: RecordingManager.Status.busy)
    
    init() {
        print("Initializing...")
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }.environmentObject(recordingManager)
    }
}

func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString),
           UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
        }
    }
