using Microsoft.AI.MachineLearning;
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Shapes;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace WinML_MoveNet
{
    /// <summary>
    /// A WinML application where an onnx model is loaded and the output evaluated
    /// relying on this repo: https://github.com/microsoft/Windows-Machine-Learning
    /// to automatically create input/output/model bindings when importing onnx model, the following
    /// VS plugin must be installed: https://marketplace.visualstudio.com/items?itemName=WinML.MLGenV2
    /// </summary>
    public sealed partial class MainPage : Page
    {
        //private MediaCapture _mediaCapture;
        private MediaFrameReader _mediaFrameReader;

        // set default values (192x192 default for moveNet lightning)
        private int _imgWidth = 192;
        private int _imgHeight = 192;            

        private model_float32_lightningModel _model;
        private model_float32_lightningInput _input = new model_float32_lightningInput();
        private model_float32_lightningOutput _output;

        private bool _debugging = false;
        private SoftwareBitmap _debugBitmap;
        private TensorizationHelper _tensorizationHelper = new TensorizationHelper();

        public MainPage()
        {
            this.InitializeComponent();

            InitModelAsync();            
        }

        // initialize onnx model
        private async Task InitModelAsync()
        {
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/model_float32_lightning.onnx"));
            _model = await model_float32_lightningModel.CreateFromStreamAsync(modelFile as IRandomAccessStreamReference);

            // initialize camera after model has been initialized
            await InitCameraAsync();
        }

        // if debugging, load image, otherwise initialize camera
        private async Task InitCameraAsync()
        {
            // if debugging, initialize and show only an image
            if (_debugging)
            {
                // this is a simple 3x5 image with well defined RGB values, allowing easier debugging of the input arrays
                //StorageFile file = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/TestImg3Wx5H.png"));

                // this is an image of a person formatted to the input size (192x192 for moveNet lightning)
                StorageFile file = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/personsquat_192x192.jpg"));

                // convert the loaded image to a SoftwareBitmap
                Stream ms = await file.OpenStreamForReadAsync();
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(ms.AsRandomAccessStream());
                _debugBitmap = await decoder.GetSoftwareBitmapAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);

                if (_debugBitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 || _debugBitmap.BitmapAlphaMode == BitmapAlphaMode.Straight)
                {
                    _debugBitmap = SoftwareBitmap.Convert(_debugBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }
                SoftwareBitmapSource source = new SoftwareBitmapSource();
                await source.SetBitmapAsync(_debugBitmap);

                // Show the loaded image on the MainPage.xaml  
                inputTestImage.Source = source;

                ProcessFrame();
            }
            // if not debugging, initialize camera stream; ensure webcam (and mic) are enabled in appxmanifest
            else
            {
                //if (_mediaCapture == null || _mediaCapture.CameraStreamState == Windows.Media.Devices.CameraStreamState.Shutdown || _mediaCapture.CameraStreamState == Windows.Media.Devices.CameraStreamState.NotStreaming)
                //{
                    MediaSourceFinder mediaSourceFinder = new MediaSourceFinder();
                    _mediaFrameReader = await mediaSourceFinder.InitMediaFrameReader();

                //WebCam.Source = _mediaCapture;
                //}
                inputTestImage.Source = new SoftwareBitmapSource();

                MediaFrameReaderStartStatus status = await _mediaFrameReader.StartAsync();

                _mediaFrameReader.FrameArrived += MediaFrameReader_FrameArrived;

                if (status == MediaFrameReaderStartStatus.Success)
                {
                    Debug.WriteLine("MediaFrameReaderStartStatus == Success");
                }
                else
                {
                    Debug.WriteLine($"MediaFrameReaderStartStatus != Success; {status}");
                }

                //if (_mediaCapture.CameraStreamState == Windows.Media.Devices.CameraStreamState.NotStreaming)
                //{
                //    await _mediaCapture.StartPreviewAsync();
                //    WebCam.Visibility = Visibility.Visible;
                //}
            }
        }

        private async Task ProcessFrame()
        {
            // model is only evaluated once for debugging picture
            if (_debugging)
            {
                // convert SoftwareBitmap to TensorFloat and bind to input
                _input.input00 = _tensorizationHelper.SoftwareBitmapToSoftwareTensor(_debugBitmap);

                // evaluate model
                _output = await _model.EvaluateAsync(new model_float32_lightningInput { input00 = _input.input00 });
                                
                // draw output
                await DrawJoints(_output.Identity);
            }           
        }

        // draw dots on a canvas
        private async Task DrawJoints(TensorFloat allJoints)
        {
            // clear canvas to draw new dots
            outputCanvas.Children.Clear();

            // initialize brush with certain color
            var brush = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 0, 0));

            // get TensorFloat output as VectorView to be able to access the values
            var allJoints_vec = allJoints.GetAsVectorView();

            // draw all dots that have high tracking confidence
            for (int i = 0; i < allJoints_vec.Count; i+=3)
            {
                // create as many dots as joints
                Ellipse ellipse = new Ellipse();
                ellipse.Margin = new Thickness(allJoints_vec[i + 1] * _imgWidth, allJoints_vec[i] * _imgHeight, 0, 0);
                ellipse.Fill = brush;
                ellipse.Width = 4;
                ellipse.Height = 4;
                ellipse.StrokeThickness = 2;

                // if tracking confidence above 0.1
                if (allJoints_vec[i + 2] > 0.1)
                {
                    // add dot to canvas
                    outputCanvas.Children.Add(ellipse);
                }
            }

            // only collect chart with confidence values for debugging
            // useful, since it quickly shows whether pose tracking was successful without the need for correct drawing
            if (_debugging)
            {
                float[] confidence = new float[17];
                int j = 0;
                for (int i = 0; i < allJoints_vec.Count; i += 3)
                {
                    confidence[j] = allJoints_vec[i + 2];
                    j += 1;
                }
            }
        }

        private bool _taskRunning = false;
        private SoftwareBitmap _backBuffer;
        private void MediaFrameReader_FrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
        {
            MediaFrameReference mediaFrameReference = sender.TryAcquireLatestFrame();
            VideoMediaFrame videoMediaFrame = mediaFrameReference?.VideoMediaFrame;
            SoftwareBitmap softwareBitmap = videoMediaFrame?.SoftwareBitmap;

            if (softwareBitmap != null)
            {
                if (softwareBitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 ||
                    softwareBitmap.BitmapAlphaMode != BitmapAlphaMode.Premultiplied)
                {
                    softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }

                // Swap the processed frame to _backBuffer and dispose of the unused image.
                softwareBitmap = Interlocked.Exchange(ref _backBuffer, softwareBitmap);
                softwareBitmap?.Dispose();

                // Changes to XAML ImageElement must happen on UI thread through Dispatcher
                var task = inputTestImage.Dispatcher.RunAsync(CoreDispatcherPriority.Normal,
                    async () =>
                    {
                        // Don't let two copies of this task run at the same time.
                        if (_taskRunning)
                        {
                            return;
                        }
                        _taskRunning = true;

                        // Keep draining frames from the backbuffer until the backbuffer is empty.
                        SoftwareBitmap latestBitmap;
                        while ((latestBitmap = Interlocked.Exchange(ref _backBuffer, null)) != null)
                        {
                            var imageSource = (SoftwareBitmapSource)inputTestImage.Source;
                            await imageSource.SetBitmapAsync(latestBitmap);
                            _input.input00 = _tensorizationHelper.SoftwareBitmapToSoftwareTensor(latestBitmap);

                            _output = await _model.EvaluateAsync(new model_float32_lightningInput { input00 = _input.input00 });

                            await DrawJoints(_output.Identity);

                            latestBitmap.Dispose();
                        }

                        _taskRunning = false;
                    });

                mediaFrameReference.Dispose();
            }
        }

        private async Task<SoftwareBitmap> CropBitmap(SoftwareBitmap softwareBitmap)
        {
            SoftwareBitmap croppedBitmap;
            using (InMemoryRandomAccessStream stream = new InMemoryRandomAccessStream())
            {
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
                // Get the SoftwareBitmap representation of the file
                croppedBitmap = await decoder.GetSoftwareBitmapAsync(decoder.BitmapPixelFormat, 
                    BitmapAlphaMode.Ignore, 
                    new BitmapTransform() { Bounds = new BitmapBounds() { X = (uint)(softwareBitmap.PixelWidth - _imgWidth) / 2, Y = (uint)(softwareBitmap.PixelHeight - _imgHeight) / 2, Width = (uint)_imgWidth, Height = (uint)_imgHeight } }, 
                    ExifOrientationMode.IgnoreExifOrientation, 
                    ColorManagementMode.DoNotColorManage);                       
                
            }
            return croppedBitmap;
        }

    }
}
