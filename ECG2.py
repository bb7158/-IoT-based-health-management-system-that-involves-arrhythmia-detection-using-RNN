import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from drawnow import drawnow
from fpdf import FPDF
from tensorflow.keras.models import load_model
import serial

# Setup the serial connection (ensure the COM port matches your setup)
ser = serial.Serial('COM6', 115200)  # Adjust the COM port as needed

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_ecg_and_arrhythmia_probability(ecg_data, arrhythmia_probabilities):
    fig, ax1 = plt.subplots()

    # Plot ECG data
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('ECG Voltage (mV)', color='tab:blue')
    ax1.plot(ecg_data, color='tab:blue', label='ECG Signal')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(100, 650)

    # Create a second y-axis for arrhythmia probability
    ax2 = ax1.twinx()
    ax2.set_ylabel('Probability of Arrhythmia', color='tab:red')
    ax2.plot(arrhythmia_probabilities, color='tab:red', linestyle='--', label='Arrhythmia Probability', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('ECG Signal and Probability of Arrhythmia')
    fig.tight_layout()
    plt.show()

def generate_report(ecg_data, bpm, arrhythmia_probabilities, images, report_name):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="ECG Analysis Report", ln=True, align='C')

    pdf.cell(200, 10, txt=f"Beats Per Minute: {bpm}", ln=True)
    pdf.cell(200, 10, txt="Arrhythmia Probabilities Summary:", ln=True)
    
    for i, prob in enumerate(arrhythmia_probabilities):
        pdf.cell(0, 10, txt=f"Time {i}: Probability {prob:.2f}", ln=True)

    pdf.add_page()
    pdf.cell(200, 10, txt="ECG Signal Snapshot", ln=True, align='C')
    pdf.image(images[0], x=10, y=30, w=180)
    
    pdf.add_page()
    pdf.cell(200, 10, txt="Distance Histogram Snapshot", ln=True, align='C')
    pdf.image(images[1], x=10, y=30, w=180)

    pdf.output(report_name)

def main():
    while True:
        try:
            min = float(input('Recording time (minutes) >> '))
            cantidad = min * 60 * 250  # Samples for the given duration
            break
        except ValueError:
            print('Please enter a valid number.')

    print('Starting recording...')
    
    # Set up real-time plotting
    plt.ion()  # Interactive mode for real-time plotting
    data = []
    start_time = time.time()  # Record the start time
    last_read_time = start_time  # Track the last read time

    plt.figure()  # Create a figure
    plt.xlabel('Time (samples)')
    plt.ylabel('Voltage (mV)')
    plt.title('Electrocardiogram')
    plt.ylim(100, 650)  # Set Y-axis limits

    while len(data) < cantidad:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()  # Read line and decode
                ecg_value = float(line)  # Convert to float
                data.append(ecg_value)
                last_read_time = time.time()  # Update last read time

                # Clear and update the plot
                plt.clf()  # Clear the current figure
                plt.plot(data, color='blue')  # Plot ECG data
                plt.xlim(0, len(data))  # Set X-axis limits
                plt.ylim(100, 650)  # Set Y-axis limits
                plt.xlabel('Time (samples)')
                plt.ylabel('Voltage (mV)')
                plt.title('Electrocardiogram')
                
                plt.pause(0.01)  # Allow GUI to update

            # Check if the specified time has elapsed or if no data has been received for 2 seconds
            elapsed_time = time.time() - start_time
            if elapsed_time >= min * 60:  # Convert minutes to seconds
                break
            if time.time() - last_read_time > 2:  # Timeout after 2 seconds of no data
                print("No data received for 2 seconds. Stopping recording.")
                break  # Stop data collection after timeout

        except ValueError:
            print("Problem capturing data", end='\n')
            guardar = input('Do you want to save the data: s = yes, n = no: ')

            if guardar.lower() == 's':
                ecg_data = pd.DataFrame(data=data, columns=['Voltage (mV)'])  # Add header to DataFrame
                nombre = input("File name: ")
                archivo = nombre + ".csv"
                ecg_data.to_csv(archivo, index_label='Index')  # Generate a CSV file with ECG data
            else:
                pass

    print('Data captured')

    # Convert list to DataFrame
    ecg_data = pd.DataFrame(data=data, columns=['Voltage (mV)'])  # Add header to DataFrame
    nombre = input("File name: ")
    archivo = nombre + ".csv"
    ecg_data.to_csv(archivo, index_label='Index')  # Generate a CSV file with ECG data

    # Use generated data directly
    ecg_data = ecg_data['Voltage (mV)'].values  # Use the column with voltage values

    # Denoise the ECG signal
    fs = 250  # Sampling frequency
    cutoff = 40  # Cutoff frequency for the low-pass filter
    ecg_data_denoised = lowpass_filter(ecg_data, cutoff, fs)

    # Detect R peaks in the denoised ECG signal
    peaks, _ = find_peaks(ecg_data_denoised, distance=150)
    distancias = np.diff(peaks)

    media = np.mean(distancias)

    # Calculate and show beats per minute (BPM)
    bpm = (ecg_data.size / media) / (ecg_data.size / 15000)

    print('Registered {} beats per minute.'.format(round(bpm)))

    # Show the graph of detected R peaks
    fig1 = plt.figure(1)
    plt.plot(ecg_data_denoised, 'b')
    plt.plot(peaks, ecg_data_denoised[peaks], 'rx')
    plt.title('Detected R Peaks')

    # Show the histogram of distances between R peaks
    fig2 = plt.figure(2)
    plt.hist(distancias)
    plt.xlabel('Distance (samples)')
    plt.ylabel('Frequency')
    plt.title('Distribution of distances between local maxima (peaks)')
    plt.show()

    # Load the pre-trained model
    model = load_model('C:/Users/Bessel Binny/Downloads/ecg_model.h5')

    # Predict arrhythmia probabilities using the loaded model
    arrhythmia_probabilities = model.predict(ecg_data_denoised.reshape(1, -1)).flatten()

    # Plot ECG data and arrhythmia probabilities
    plot_ecg_and_arrhythmia_probability(ecg_data_denoised, arrhythmia_probabilities)

    # Save generated images
    guardar = input('Save images = s, do not save = n: ')

    if guardar.lower() == 's':
        fig1.savefig(nombre + "ecg.png")
        fig2.savefig(nombre + "dist.png")
        
        # Generate report
        report_name = nombre + "_ECG_Report.pdf"
        generate_report(ecg_data, round(bpm), arrhythmia_probabilities, [nombre + "ecg.png", nombre + "dist.png"], report_name)
        print(f'Report saved as {report_name}')
    else:
        pass

# Call the main function
if __name__ == '__main__':
    main()
