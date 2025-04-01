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
            Text("Please make the buttons and text large and accessible.")
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
                
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
