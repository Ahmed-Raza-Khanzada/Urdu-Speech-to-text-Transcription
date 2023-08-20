import pyaudio
import numpy as np
import socketio


# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize SocketIO client
sio = socketio.Client()


# Define a callback function to stream audio data
def audio_callback(in_data, frame_count, time_info, status):
    # Convert audio data to NumPy array
    audio_buffer = np.frombuffer(in_data, dtype=np.int16)
    # Emit audio data to server
    sio.emit('audio_data', audio_buffer.tobytes())
    # Return None as we are not using any output stream
    return None, pyaudio.paContinue


# Define the main function
def main():
    # Connect to the server
    @sio.event
    def connect():
        print('Connected to server')

    # Handle server disconnects
    @sio.event
    def disconnect():
        print('Disconnected from server')
        sio.disconnect()
    
    # Connect to the server
    sio.connect('http://127.0.0.1:5031')
    connect()

    # Open the default input stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    # rate=44100,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024,
                    stream_callback=audio_callback)
    
    # Start the stream
    stream.start_stream()
    
    # @sio.on('my_emittting')#, namespace='/')
    @sio.on('audio_dt')#, namespace='/')
    def handle_audio_data(audio_data):
        print(audio_data)
    # Wait for the stream to finish
    while stream.is_active():
        
        pass
    
    # Stop the stream
    stream.stop_stream()
    stream.close()
    
    # Terminate PyAudio
    p.terminate()


# Call the main function
if __name__ == '__main__':
    main()
