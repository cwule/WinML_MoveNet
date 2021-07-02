using System;
using System.Linq;

using Windows.Media.Capture.Frames;
using Windows.Media.Capture;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace WinML_MoveNet
{
    class MediaSourceFinder
    {
        public MediaCapture _mediaCapture;
        MediaFrameReader _mediaFrameReader;
        MediaFrameSource _mediaFrameSource;

        private bool _profileFound = false;
        public async Task<MediaFrameReader> InitMediaFrameReader()
        {
            /// Select frame sources and frame source groups                     
            var sourceGroups = await MediaFrameSourceGroup.FindAllAsync();
            MediaCaptureInitializationSettings settings = null;

            foreach (MediaFrameSourceGroup sg in sourceGroups)
            {
                // Find videoprofile that supports videoconferencing (contains lowest res profile in HL2)
                IReadOnlyList<MediaCaptureVideoProfile> profileList = MediaCapture.FindKnownVideoProfiles(
                                          sg.Id,
                                          KnownVideoProfile.VideoConferencing);

                foreach (MediaCaptureVideoProfile profile in profileList)
                {
                    // ensure to have correct description, eg. SupportedRecordMediaDescription, otherwise some profiles not available
                    IReadOnlyList<MediaCaptureVideoProfileMediaDescription> recordMediaDescription =
                        profile.SupportedRecordMediaDescription;
                    foreach (MediaCaptureVideoProfileMediaDescription videoProfileMediaDescription in recordMediaDescription)
                    {
                        // we want the lowest possible HL2 camera resolution (closes to what is fed in MoveNet)
                        if (videoProfileMediaDescription.Height == 240)
                        {
                            settings = new MediaCaptureInitializationSettings()
                            {
                                SourceGroup = sg,
                                VideoProfile = profile,
                                SharingMode = MediaCaptureSharingMode.ExclusiveControl,
                                MemoryPreference = MediaCaptureMemoryPreference.Cpu,
                                StreamingCaptureMode = StreamingCaptureMode.Video
                            };
                            _profileFound = true;
                            break;
                        }
                    }
                    if (_profileFound)
                    {
                        // Profile found. Stop looking.
                        break;
                    }
                }
            }

            if (settings == null)
            {
                settings = new MediaCaptureInitializationSettings()
                {
                    SharingMode = MediaCaptureSharingMode.ExclusiveControl,
                    MemoryPreference = MediaCaptureMemoryPreference.Cpu,
                    StreamingCaptureMode = StreamingCaptureMode.Video
                };
            }

            /// Initialize the MediaCapture object to use the selected frame source group
            _mediaCapture = new MediaCapture();

            try
            {
                await _mediaCapture.InitializeAsync(settings);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("MediaCapture initialization failed: " + ex.Message);
                return null;
            }

            MediaFrameFormat preferredFormat = _mediaCapture.FrameSources[sourceGroups.FirstOrDefault().SourceInfos[0].Id].SupportedFormats.FirstOrDefault();

            // there are multiple sourcegroups (the one I need is the first, could also be a different one though, and each sourcegroup has multiple framesources
            // that's why all these loops are necessary to find the right one
            // test, whether this can also be just prescribed directly
            foreach (var src in sourceGroups.FirstOrDefault().SourceInfos)
            {
                _mediaFrameSource = _mediaCapture.FrameSources[src.Id];

                var supportedFormats = _mediaFrameSource.SupportedFormats;
                foreach (MediaFrameFormat format in supportedFormats)
                {
                    var height = format.VideoFormat.Height;
                    var fr = format.FrameRate.Numerator;
                    if (height == 240 && fr == 15)
                    {
                        preferredFormat = format;
                        break;
                    }
                }
            }

            await _mediaFrameSource.SetFormatAsync(preferredFormat);

            /// Create a frame reader for the frame source
            _mediaFrameReader = await _mediaCapture.CreateFrameReaderAsync(_mediaFrameSource);                      

            return _mediaFrameReader;
        }

    }
}
