//
//  ContentView.swift
//  Local STT Model Runner
//
//  Created by Ty DeVito on 3/31/25.
//

import SwiftUI

struct ContentView: View {
    @State var isRecording : Bool = false;
    var body: some View {
        VStack {
            
            Text("Text to speech app")
                .padding().bold()
            ScrollView(.vertical) {
                Text("Press Record to Begin...")
                    .font(.system(size: 35)).bold()
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
            Button(action: {
                Test()
                isRecording.toggle()
            }) {
                HStack {
                    Image(systemName: isRecording ? "stop.fill" : "microphone").bold()
                        .font(.system(size: 40))
                    Text(isRecording ? "Stop" : "Record")
                        .font(.system(size: 40))
                        .bold()
                    
                }.padding()
                    .frame(maxWidth: .infinity, minHeight: 150)
            }.buttonStyle(.borderedProminent).tint(isRecording ? .red : .blue)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
