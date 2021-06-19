using System;
using System.Collections.Generic;
using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.UI.Xaml.Media.Imaging;


/// <summary>
/// This class reorders SoftwareBitmaps into Tensorfloats
/// convert from BGR8 to RGB8 and the other way round
/// convert from NCWH to NWHC and NCWH to NHWC
/// </summary>

namespace WinML_MoveNet
{
    public class TensorizationHelper
    {
        private bool _inputNormalized = false;

        /// <summary>
        /// default, don't normalize input
        /// </summary>
        /// <param name="softwareBitmap"></param>
        /// <returns></returns>
        public TensorFloat SoftwareBitmapToSoftwareTensor(SoftwareBitmap softwareBitmap, string outputType)
        {
            // Manually tensorize from CPU resource, steps:
            // 1. Get the access to buffer of softwarebitmap
            // 2. Transform the data in buffer to a vector of float

            // cast Byte array to IEnumerable<float> and create TensorFloat, does not resize input, ensure correct shape for onnx model input
            //TensorFloat input = TensorFloat.CreateFromIterable(new long[] { 1, softwareBitmap.PixelHeight, softwareBitmap.PixelWidth, 3 }, CreateTensor(softwareBitmap, outputType));
            long[] shape = new long[4];
            shape[0] = 1;
            shape[1] = softwareBitmap.PixelHeight;
            shape[2] = softwareBitmap.PixelWidth;
            shape[3] = 3;
            TensorFloat input = TensorFloat.CreateFromArray(new long[] { 1, 192, 192, 3 }, ConvertFrameToFloatArray(softwareBitmap));

            return input;
        }


        /// allows to normalize the input
        public TensorFloat SoftwareBitmapToSoftwareTensor(SoftwareBitmap softwareBitmap, string outputType, bool normalizeInput)
        {
            _inputNormalized = normalizeInput;
            // Manually tensorize from CPU resource, steps:
            // 1. Get the access to buffer of softwarebitmap
            // 2. Transform the data in buffer to a vector of float

            // cast Byte array to IEnumerable<float> and create TensorFloat, does not resize input, ensure correct shape for onnx model input
            TensorFloat input = TensorFloat.CreateFromIterable(new long[] { 1, softwareBitmap.PixelHeight, softwareBitmap.PixelWidth, 3 }, CreateTensor(softwareBitmap, outputType));

            return input;
        }

        private IEnumerable<float> CreateTensor(SoftwareBitmap softwareBitmap, string outputType)
        {
            switch (outputType)
            {
                case "BGR8":
                    return TensorBgrNHWC(softwareBitmap);
                case "RGB8":
                    return TensorRgbNHWC(softwareBitmap);
                case "RGB8NHWC":
                    return TensorRgbNCWH2NHWC(softwareBitmap);
                default:
                    return TensorRgbNCWH2NHWC(softwareBitmap);
            }
        }

        /// delivers byte array ordered eg. for 3 channel input image (c,x,y) = (4,3,5) :
        /// c0 x0 y0, c1 x0 y0, c2 x0 y0, c3 x0 y0, c0 x1 y0, c1 x1 y0, c2 x1 y0, c3 x1 y0, c0 x2 y0, ...
        /// tested with TestImg3Wx5H
        private byte[] ConvertFrameToByteArray(SoftwareBitmap bitmap)
        {
            byte[] bytes;
            WriteableBitmap newBitmap = new WriteableBitmap(bitmap.PixelWidth, bitmap.PixelHeight);
            bitmap.CopyToBuffer(newBitmap.PixelBuffer);


            //using (Stream stream = newBitmap.PixelBuffer.AsStream())
            //using (MemoryStream memoryStream = new MemoryStream())
            //{
            //    stream.CopyTo(memoryStream);
            //    bytes = memoryStream.ToArray();
            //}

            // this single line delivers the same output as the paragraph abve using memoryStream
            bytes = newBitmap.PixelBuffer.ToArray();
            return bytes;
        }


        /// delivers byte array ordered eg. for 3 channel input image (c,x,y) = (4,3,5) :
        /// c0 x0 y0, c1 x0 y0, c2 x0 y0, c3 x0 y0, c0 x1 y0, c1 x1 y0, c2 x1 y0, c3 x1 y0, c0 x2 y0, ...
        /// tested with TestImg3Wx5H
        private float[] ConvertFrameToFloatArray(SoftwareBitmap bitmap)
        {
            byte[] bytes;
            WriteableBitmap newBitmap = new WriteableBitmap(bitmap.PixelWidth, bitmap.PixelHeight);
            bitmap.CopyToBuffer(newBitmap.PixelBuffer);

            // this single line delivers the same output as the paragraph abve using memoryStream
            bytes = newBitmap.PixelBuffer.ToArray();
            List<float> float_arr = new List<float>();

            // drop the alpha channel
            int a = 0;
            int j = 0;
            for (int i = 0; i < bytes.Length; i++)
            {
                if (a == 3)
                {
                    a = 0;
                }
                else
                {
                    float_arr.Add(Convert.ToSingle(bytes[i]));
                    a += 1;
                }
            }

            return float_arr.ToArray();
        }

        /// Transform a SoftwareBitmap to a Tensorfloat; 
        /// adapted from https://github.com/microsoft/Windows-Machine-Learning/blob/b803e660222bca0c0121df3dbd54c66021de31a7/Samples/CustomTensorization/CustomTensorization/TensorConvertor.cpp#L77
        /// Reorder BGR8 or RGB8 NCWH (or NCHW) to BGR8 NWHC (or NCHW) (and drop alpha channel)
        /// does not swap width and height        
        public IEnumerable<float> TensorBgrNHWC(SoftwareBitmap softwareBitmap)
        {
            // 1. Get the access to buffer of softwarebitmap
            byte[] bytes = ConvertFrameToByteArray(softwareBitmap);

            // The channels of image stored in buffer is in order of BGRA-BGRA-BGRA-BGRA. 
            // Then we transform it to the order of BBBBB....GGGGG....RRRR....AAAA(dropped)
            BitmapPixelFormat pixelformat = softwareBitmap.BitmapPixelFormat;
            int size = bytes.Length;

            // if input should be normalized between 0 and 1
            if (_inputNormalized)
            {
                // the current BitmapPixelFormat of the bitmap decides the starting index i of the for-loops
                // for BGR8, index i = 0 is B, 1 is G, 2 is R, 3 is alpha(dropped).
                // depending on starting value of i, we start with a different channel
                // the order of the for-loops has to correspond to desired output, eg. first b-loop, then g-loop, then r-loop
                if (pixelformat == BitmapPixelFormat.Bgra8)
                {
                    for (int i = 0; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i] / 255.0);
                        yield return b;
                    }
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i] / 255.0);
                        yield return g;
                    }
                    for (int i = 2; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i] / 255.0);
                        yield return r;
                    }
                }
                else if (pixelformat == BitmapPixelFormat.Rgba8)
                {
                    for (int i = 2; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i] / 255.0);
                        yield return b;
                    }
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i] / 255.0);
                        yield return g;
                    }
                    for (int i = 0; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i] / 255.0);
                        yield return r;
                    }
                }
            }
            // if input does not need to be normalized
            else
            {
                if (pixelformat == BitmapPixelFormat.Bgra8)
                {
                    for (int i = 0; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i]);
                        yield return b;
                    }
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i]);
                        yield return g;
                    }
                    for (int i = 2; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i]);
                        yield return r;
                    }
                }
                else if (pixelformat == BitmapPixelFormat.Rgba8)
                {
                    for (int i = 2; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i]);
                        yield return b;
                    }
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i]);
                        yield return g;
                    }                    
                    for (int i = 0; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i]);
                        yield return r;
                    }
                }
            }
        }

        /// Transform a SoftwareBitmap to a Tensorfloat; 
        /// adapted from https://github.com/microsoft/Windows-Machine-Learning/blob/b803e660222bca0c0121df3dbd54c66021de31a7/Samples/CustomTensorization/CustomTensorization/TensorConvertor.cpp#L77
        /// Reorder BGR8 or RGB8 NCWH (or NCHW) to RGB8 NWHC (or NCHW) (and drop alpha channel)
        /// does not swap width and height 
        public IEnumerable<float> TensorRgbNHWC(SoftwareBitmap softwareBitmap)
        {
            // 1. Get the access to buffer of softwarebitmap
            byte[] bytes = ConvertFrameToByteArray(softwareBitmap);

            // The channels of image stored in buffer is in order of BGRA-BGRA-BGRA-BGRA. 
            // Then we transform it to the order of RRRRR....GGGGG....BBBBB....AAAA(dropped)
            BitmapPixelFormat pixelformat = softwareBitmap.BitmapPixelFormat;
            int size = bytes.Length;

            // if input should be normalized between 0 and 1
            if (_inputNormalized)
            {
                // the current BitmapPixelFormat of the bitmap decides the starting index i of the for-loops
                // for BGR8, index i = 0 is B, 1 is G, 2 is R, 3 is alpha(dropped).
                // depending on starting value of i, we start with a different channel
                // the order of the for-loops has to correspond to desired output, eg. first r-loop, then g-loop, then b-loop
                if (pixelformat == BitmapPixelFormat.Bgra8)
                {
                    for (int i = 2; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i] / 255.0);
                        yield return r;
                    }                    
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i] / 255.0);
                        yield return g;
                    }
                    for (int i = 0; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i] / 255.0);
                        yield return b;
                    }
                }
                else if (pixelformat == BitmapPixelFormat.Rgba8)
                {
                    for (int i = 0; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i] / 255.0);
                        yield return r;
                    }
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i] / 255.0);
                        yield return g;
                    }
                    for (int i = 2; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i] / 255.0);
                        yield return b;
                    }
                }
            }
            // if not normalized
            else
            {
                if (pixelformat == BitmapPixelFormat.Bgra8)
                {
                    for (int i = 2; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i]);
                        yield return r;
                    }
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i]);
                        yield return g;
                    }
                    for (int i = 0; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i]);
                        yield return b;
                    }
                }
                else if (pixelformat == BitmapPixelFormat.Rgba8)
                {
                    for (int i = 0; i < size; i += 4)
                    {
                        var r = Convert.ToSingle(bytes[i]);
                        yield return r;
                    }
                    for (int i = 1; i < size; i += 4)
                    {
                        var g = Convert.ToSingle(bytes[i]);
                        yield return g;
                    }
                    for (int i = 2; i < size; i += 4)
                    {
                        var b = Convert.ToSingle(bytes[i]);
                        yield return b;
                    }
                }
            }
        }

        /// Transform a SoftwareBitmap to a Tensorfloat; 
        /// adapted from https://github.com/microsoft/Windows-Machine-Learning/blob/b803e660222bca0c0121df3dbd54c66021de31a7/Samples/CustomTensorization/CustomTensorization/TensorConvertor.cpp#L77
        /// Reorder RGB8 NCWH to RGB8 NHWC (Width and height switched!) (and drop alpha channel)
        public IEnumerable<float> TensorRgbNCWH2NHWC(SoftwareBitmap softwareBitmap)
        {
            // 1. Get the access to buffer of softwarebitmap
            byte[] bytes = ConvertFrameToByteArray(softwareBitmap);

            // The channels of image stored in buffer is in order of BGRA-BGRA-BGRA-BGRA. 
            /// b x0 y0, g x0 y0, r x0 y0, a x0 y0, b x1 y0, g x1 y0, r x1 y0, a x1 y0, b x2 y0, ...
            /// to
            /// y0 x0 r, y1 x0 r, y2 x0 r, .... y0 x1 r, y1 x1 r, y2 x1 r, ....            
            BitmapPixelFormat pixelformat = softwareBitmap.BitmapPixelFormat;
            int size = bytes.Length;

            if (_inputNormalized)
            {
                if (pixelformat == BitmapPixelFormat.Bgra8)
                {
                    // increment by width * channel to first get all the height values y
                    // then increment the width x by 1 to go to the next column
                    // the starting index of y depends on the channel and the input BitmapPixelFormat
                    // for BGR8, the starting index of b is y = 0, for g is y = 1, and for r is y = 2
                    // the for-loops are ordered according to the desired output, eg. r-loop, g-loop, b-loop
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {                        
                        for (int y = 2; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var r = Convert.ToSingle(bytes[y + x * 4] / 255.0);
                            yield return r;
                        }
                    }
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 1; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var g = Convert.ToSingle(bytes[y + x * 4] / 255.0);
                            yield return g;
                        }
                    }
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        // starting values of y decides, which channel is first, 0=b 1=g 2=r
                        for (int y = 0; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var b = Convert.ToSingle(bytes[y + x * 4] / 255.0);
                            yield return b;
                        }
                    }
                }
                else if (pixelformat == BitmapPixelFormat.Rgba8)
                {
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 0; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var r = Convert.ToSingle(bytes[y + x * 4] / 255.0);
                            yield return r;
                        }
                    }
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 1; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var g = Convert.ToSingle(bytes[y + x * 4] / 255.0);
                            yield return g;
                        }
                    }                    
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 2; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var b = Convert.ToSingle(bytes[y + x * 4] / 255.0);
                            yield return b;
                        }
                    }
                }
            }
            else
            {
                // suppose the model expects RGB image.
                // index 0 is B, 1 is G, 2 is R, 3 is alpha(dropped).
                if (pixelformat == BitmapPixelFormat.Bgra8)
                {
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 2; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var r = Convert.ToSingle(bytes[y + x * 4]);
                            yield return r;
                        }
                    }
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 1; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var g = Convert.ToSingle(bytes[y + x * 4]);
                            yield return g;
                        }
                    }                    
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 0; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var b = Convert.ToSingle(bytes[y + x * 4]);
                            yield return b;
                        }
                    }
                }
                else if (pixelformat == BitmapPixelFormat.Rgba8)
                {
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 0; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var r = Convert.ToSingle(bytes[y + x * 4]);
                            yield return r;
                        }
                    }
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 1; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var g = Convert.ToSingle(bytes[y + x * 4]);
                            yield return g;
                        }
                    }                    
                    for (int x = 0; x < softwareBitmap.PixelWidth; x += 1)
                    {
                        for (int y = 2; y < size; y += softwareBitmap.PixelWidth * 4)
                        {
                            var b = Convert.ToSingle(bytes[y + x * 4]);
                            yield return b;
                        }
                    }
                }
            }
        }

    }
}
