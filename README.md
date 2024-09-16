# Face Cropping API

This project provides a simple HTTP server that offers face detection and cropping functionality through an API endpoint. It uses TensorFlow.js and face-api.js for face detection, and Sharp for image processing.

## Features

- Face detection in images
- Cropping images to focus on detected faces
- Customizable output dimensions and padding
- Simple HTTP GET endpoint for easy integration

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Node.js (v22 or later recommended)
- npm (usually comes with Node.js)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/legalease/facecrop.git
   cd facecrop
   ```

2. Install the dependencies:
   ```
   npm install
   ```

## Usage

1. Start the server:
   ```
   npm run start
   ```

2. The server will start running on `http://localhost:8080`

3. To use the face cropping API, send a GET request to the `/faceCrop` endpoint with the following query parameters:
   - `width`: The desired width of the output image (required)
   - `height`: The desired height of the output image (required)
   - `padding`: The padding around the detected face (optional, default is 0.1)
   - `image`: The URL of the input image (required)

   Example:
   ```
   http://localhost:8080/faceCrop?width=300&height=300&padding=0.2&image=https://example.com/image.jpg
   ```

4. The API will return the cropped image focused on the detected face, or an error message if no face is detected or if there's an issue processing the image.
