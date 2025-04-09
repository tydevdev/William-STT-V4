//
//  ContentView.swift
//  William_STT_Runner
//
//  Created by Ty DeVito on 3/31/25.
//

import SwiftUI
import AVFAudio

struct ContentView: View {
    @EnvironmentObject var status : RecordingManager
    var body: some View {
        VStack {
            
            Text("Text to speech prototype")
                .padding().bold()
            ScrollView(.vertical) {
                Text(status.transcription)
                    .font(.system(size: 35)).bold()
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
            Button(action: {
                status.OnPressButton()
            }) {
                HStack {
                    Image(systemName: status.buttonIcon).bold()
                        .font(.system(size: 40))
                    Text(status.buttonLabel)
                        .font(.system(size: 40))
                        .bold()
                    
                }.padding()
                    .frame(maxWidth: .infinity, minHeight: 150)
            }.buttonStyle(.borderedProminent)
                .tint(status.buttonColor)
                .disabled(status.buttonDisabled)
        }
        .padding()
        
        .alert(isPresented: $status.showMicrophonePermissionAlert) {
                    Alert(
                        title: Text("Microphone Access Denied"),
                        message: Text("Please enable microphone access in Settings to record audio."),
                        primaryButton: .default(Text("Settings"), action: openSettings),
                        secondaryButton: .cancel()
                    )
                }
    }
}

#Preview {
    ContentView().environmentObject(RecordingManager(status: RecordingManager.Status.busy))
}
