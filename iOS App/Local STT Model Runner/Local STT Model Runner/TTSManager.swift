//
//  TTSManager.swift
//  Local STT Model Runner
//
//  Created by Dari Gomez on 4/10/25.
//

import Foundation

class TTSManager {
    init() {
//        Perform any setup needed here
    }
    
//    This function is called when audio recording finishes and the model should begin
    func beginTranscription(_ filePath : String, callback : (String) -> Void) {
        
        
    //    Keep track of the callback variable and
    //    call callback(result: String) when done
        callback("Test")
    }
}
