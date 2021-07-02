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
        int inputWidth = 192;
        int inputHeight = 192;
        /// <summary>
        /// default, don't normalize input
        /// </summary>
        /// <param name="softwareBitmap"></param>
        /// <returns></returns>
        public TensorFloat SoftwareBitmapToSoftwareTensor(SoftwareBitmap softwareBitmap)
        {
            // Manually tensorize from CPU resource, steps:
            // 1. Get the access to buffer of softwarebitmap
            // 2. Transform the data in buffer to a vector of float

            // convert float array to TensorFloat, does not resize input, ensure correct shape for onnx model input            
            //TensorFloat input = TensorFloat.CreateFromArray(new long[] { 1, 192, 192, 3 }, ConvertFrameToFloatArray(softwareBitmap));
            TensorFloat input = TensorFloat.CreateFromArray(new long[] { 1, inputWidth, inputHeight, 3 }, ConvertCropFrameToFloatArray(softwareBitmap, inputWidth, inputHeight));

            return input;
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

        /// delivers byte array ordered eg. for 3 channel input image (c,x,y) = (4,3,5) :
        /// c0 x0 y0, c1 x0 y0, c2 x0 y0, c3 x0 y0, c0 x1 y0, c1 x1 y0, c2 x1 y0, c3 x1 y0, c0 x2 y0, ...
        /// tested with TestImg3Wx5H
        /// cus to desired size
        private float[] ConvertCropFrameToFloatArray(SoftwareBitmap bitmap, int inputWidth, int inputHeight)
        {
            int bmWidth = bitmap.PixelWidth;
            int bmHeight = bitmap.PixelHeight;

            // the value that will be cropped of both top and bottom
            int cropWidth = (bmWidth - inputWidth) / 2;
            int cropHeight = (bmHeight - inputHeight) / 2;

            byte[] bytes;
            WriteableBitmap newBitmap = new WriteableBitmap(bitmap.PixelWidth, bitmap.PixelHeight);
            bitmap.CopyToBuffer(newBitmap.PixelBuffer);

            // this single line delivers the same output as the paragraph abve using memoryStream
            bytes = newBitmap.PixelBuffer.ToArray();
            List<float> float_arr = new List<float>();

            int cropTop = 4 * bmWidth * cropHeight;
            int cropBottom = bytes.Length - cropTop;
            // drop the alpha channel
            int a = 0;
            int j = 0;
            // skip the first rows (cropheight) and in the first row that counts, skip first columns on left hand cropping side
            // only convert all the values till the last relevant values are reached
            for (int i = cropTop; i < cropBottom; i++)
            {                
                // if index in the left hand cropping region
                if (i < 4 * cropWidth + cropTop + 4 * bmWidth * j)
                {
                    if (a == 3)
                        a = -1;
                }
                // if index in the right hand cropping region
                else if (i >= 4 * (cropWidth + inputWidth) + cropTop + 4 * bmWidth * j)
                {
                    if (a == 3)
                        a = -1;
                    // increment at end of each row
                    if (i % (4 * bmWidth) == 0)
                        j += 1;
                }                
                else
                {
                    if (a == 3)
                        a = -1;
                    else
                        float_arr.Add(Convert.ToSingle(bytes[i]));                 
                }
                a += 1;                
            }
            return float_arr.ToArray();
        }
    }
}
