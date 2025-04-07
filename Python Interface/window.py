from tkinter import *

# Create the main window
root = Tk()
root.title("Simple Tkinter App")
root.geometry("300x200")

# Create a label
label = Label(root, text="Hello, Tkinter!")
label.pack(pady=20)

recording = False

def record():
    print("Recording... (PLACEHOLDER)")
    # Start recording audio here
    # You would typically use a library like pyaudio or sounddevice to record audio (this is what copilot told me???)

def stop_record():
    print("Stopped recording. (PLACEHOLDER)")
    # Here you would add the code to stop the recording and save the file

# Create a button
def on_button_click():
    global recording
    if (not recording):
        label.config(text="Now recording... (PLACEHOLDER)")
        record()
    else:
        label.config(text="Click to record again (PLACEHOLDER)")
        stop_record()
    recording = not recording


button = Button(root, text="Click Here :D", command=on_button_click)
############ start recording on button click
## records until button (says stop) is pressed
############ save audio as .wav with good name
############ NEVER OVERWRITE FILES, store them in another folder (like one folder for all audio files)
## call main function (with file name parameter) from model reference example. py
## need relative reference for variablke audioFile^^^^^^^in that file not absolute path w/ tydevito
## main function should return result and // print result onto screen
## copy button, later



button.pack(pady=10)


# Add a copyable text box at the bottom
def update_text_box(content):
    text_box.delete(1.0, END)  # Clear the text box
    text_box.insert(END, content)  # Insert new content

text_box = Text(root, height=2, wrap=WORD)
text_box.insert(END, "Your result will appear here.")
text_box.config(state="normal")  # Allow copying but prevent editing
text_box.pack(pady=10)

# # Add a "Copy" button /// seems to be broken rn causing inf loop, i need to make window bigger also
# def copy_to_clipboard():
#     root.clipboard_clear()  # Clear the clipboard
#     root.clipboard_append(text_box.get(1.0, END).strip())  # Copy text from the text box
#     root.update()  # Update the clipboard content
#     print("Text copied to clipboard!")  # Optional feedback

# copy_button = Button(root, text="Copy Text", command=copy_to_clipboard)
# copy_button.pack(pady=10)

# Example: Update the text box with some result
# update_text_box("This is the result of your operation.")


# Run the main event loop
root.mainloop()