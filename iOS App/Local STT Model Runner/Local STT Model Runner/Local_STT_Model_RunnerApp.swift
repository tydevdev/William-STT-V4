//
//  Local_STT_Model_RunnerApp.swift
//  Local STT Model Runner
//
//  Created by Ty DeVito on 3/31/25.
//

import SwiftUI


@main
struct Local_STT_Model_RunnerApp: App {
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
