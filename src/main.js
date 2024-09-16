const http = require('http');
const url = require('url');
const image = require('@canvas/image');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const faceapi = require('../model/face-api.node.js');

const modelPath = './model';
const ssdOptions = { minConfidence: 0.1, maxResults: 10 };

async function cropFace(inputBuffer, width, height, padding = 0.1) {
  // Process the image
  const canvas = await image.imageFromBuffer(inputBuffer);
  const imageData = image.getImageData(canvas);

  // Create tensor from image data
  const tensor = tf.tidy(() => {
    const data = tf.tensor(Array.from(imageData?.data || []), [canvas.height, canvas.width, 4], 'int32');
    const channels = tf.split(data, 4, 2);
    const rgb = tf.stack([channels[0], channels[1], channels[2]], 2);
    return tf.reshape(rgb, [1, canvas.height, canvas.width, 3]);
  });

  // Load face detection model
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);

  // Detect faces
  const optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options(ssdOptions);
  const result = await faceapi.detectAllFaces(tensor, optionsSSDMobileNet);

  if (result.length === 0) {
    console.error('No faces detected');
    return null;
  }

  // Get the first detected face
  const detection = result[0];
  const box = detection._box;

  // Calculate the extended box with padding
  const paddingWidth = box._width * padding;
  const paddingHeight = box._height * padding;

  const extendedBox = {
    left: Math.max(0, Math.round(box._x - paddingWidth / 2)),
    top: Math.max(0, Math.round(box._y - paddingHeight / 2)),
    width: Math.round(box._width + paddingWidth),
    height: Math.round(box._height + paddingHeight)
  };

  // Ensure the box doesn't exceed image dimensions
  extendedBox.width = Math.min(extendedBox.width, canvas.width - extendedBox.left);
  extendedBox.height = Math.min(extendedBox.height, canvas.height - extendedBox.top);

  // Calculate aspect ratios
  const boxAspectRatio = extendedBox.width / extendedBox.height;
  const targetAspectRatio = width / height;

  // Adjust crop box to maintain aspect ratio
  if (boxAspectRatio > targetAspectRatio) {
    // Box is wider, adjust height
    const newHeight = extendedBox.width / targetAspectRatio;
    const heightDiff = newHeight - extendedBox.height;
    extendedBox.top = Math.max(0, Math.round(extendedBox.top - heightDiff / 2));
    extendedBox.height = Math.round(newHeight);
  } else {
    // Box is taller, adjust width
    const newWidth = extendedBox.height * targetAspectRatio;
    const widthDiff = newWidth - extendedBox.width;
    extendedBox.left = Math.max(0, Math.round(extendedBox.left - widthDiff / 2));
    extendedBox.width = Math.round(newWidth);
  }

  // Ensure the box doesn't exceed image dimensions (again, after aspect ratio adjustment)
  extendedBox.width = Math.min(extendedBox.width, canvas.width - extendedBox.left);
  extendedBox.height = Math.min(extendedBox.height, canvas.height - extendedBox.top);

  // Crop and resize the face thumbnail
  const croppedBuffer = await sharp(inputBuffer)
    .extract(extendedBox)
    .resize(width, height, { fit: 'cover' })
    .toBuffer();

  return croppedBuffer;
}

const server = http.createServer(async (req, res) => {
  const parsedUrl = url.parse(req.url, true);
  
  if (parsedUrl.pathname === '/faceCrop' && req.method === 'GET') {
    const { width, height, padding, image: imageUrl } = parsedUrl.query;

    if (!width || !height || !imageUrl) {
      res.writeHead(400, { 'Content-Type': 'text/plain' });
      res.end('Missing required parameters');
      return;
    }

    try {
      // Download the image
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const inputBuffer = await response.arrayBuffer();

      // Process the image
      const outputBuffer = await cropFace(Buffer.from(inputBuffer), parseInt(width), parseInt(height), parseFloat(padding) || 0.1);

      if (!outputBuffer) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('No face detected in the image');
        return;
      }

      // Send the processed image
      res.writeHead(200, { 'Content-Type': 'image/jpeg' });
      res.end(outputBuffer);
    } catch (error) {
      console.error('Error processing image:', error);
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Error processing image');
    }
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not Found');
  }
});

const PORT = 8080;
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
