//
//  ContentView.swift
//  Local STT Model Runner
//
//  Created by Ty DeVito on 3/31/25.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            
            Text("Future transcript over here!")
                 .padding()
            Image(systemName: "microphone")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, friend.")
            Text("William uses an iPhone 16 Pro Max")
            Text("Please make the buttons and text large and accessible.").frame(maxHeight: .infinity)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
            Button(action: {
            }) {
                Text("Record")
                    .padding()
                    .frame(maxWidth: .infinity)
                    .font(.title)
                    .bold()
            }.buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
