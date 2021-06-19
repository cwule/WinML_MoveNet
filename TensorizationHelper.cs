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
            TensorFloat input = TensorFloat.CreateFromArray(new long[] { 1, 192, 192, 3 }, ConvertFrameToFloatArray(softwareBitmap));

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
    }
}
