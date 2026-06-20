using Android.Content;
using Android.Media;
using System;
using System.Collections.Generic;
using System.IO;

namespace MindMove
{
    public class Sound
    {
        SoundDataSource mDataHitDormantMine;
        SoundDataSource mDataOrbCapture;
        SoundDataSource mDataMenuItem;
        SoundDataSource mDataRoundComplete;
        SoundDataSource mDataRoundCompleteWin;
        SoundDataSource mDataRoundEndEarly;

        MediaPlayerCycle mMediaPlayers = null;
        const int NumMediaPlayers = 5;

        public const int SoundOff = 0;
        public const int SoundEffectsOnly = 1;
        public const int MaxSoundVersion = SoundEffectsOnly;
        int mSoundVersion = MaxSoundVersion; 

        const string SoundModeFile = "SavedSoundModeV4.xml";

        Random rnd = new Random();

        public Sound()
        {
            try
            {
                mMediaPlayers = new MediaPlayerCycle(NumMediaPlayers);

                mSoundVersion = MainActivity.GetSavedIntWithDefaultValue(SoundEffectsOnly, SoundModeFile); 
                if (mSoundVersion > MaxSoundVersion || mSoundVersion < SoundOff)
                {
                    mSoundVersion = 0;
                }
                else
                {
                    LoadSoundEffects();
                }
            }
            catch { }
        }

        public void SetSoundVersion(int version)
        {
            if (version >= SoundOff && version <= MaxSoundVersion)
            {
                mSoundVersion = version;
                MainActivity.SaveIntValue(mSoundVersion, SoundModeFile);
                LoadSoundEffects();
            }
        }

        public int GetSoundVersion()
        {
            return mSoundVersion;
        }

        void LoadSoundEffects()
        {
            if (mSoundVersion >= SoundEffectsOnly)
            {
                LoadSingleSoundEffect("HitDormantMine.wav", ref mDataHitDormantMine, 1024 * 157);
                LoadSingleSoundEffect("MenuItem.wav", ref mDataMenuItem, 1024 * 114);
                LoadSingleSoundEffect("OrbCapture.wav", ref mDataOrbCapture, 1024 * 157);
                LoadSingleSoundEffect("RoundComplete.wav", ref mDataRoundComplete, 1024 * 320);
                LoadSingleSoundEffect("RoundCompleteWin.wav", ref mDataRoundCompleteWin, 1024 * 312);
                LoadSingleSoundEffect("RoundEndEarly.wav", ref mDataRoundEndEarly, 1024 * 264);
            }
        }

        void LoadSingleSoundEffect(string name, ref SoundDataSource sound, int size)
        {
            // May be a crash culprit on some devices.  Wrapping in a try-catch.
            try
            {
                var stream = new BinaryReader(MainActivity.Current.Assets.Open("Sounds1/" + name));
                byte[] soundBytes = stream.ReadBytes(size);
                stream.Close();
                stream.Dispose();
                stream.Close();

                sound = new SoundDataSource(soundBytes);
            }
            catch { }
        }

        void StartSingleSoundEffect(SoundDataSource sound)
        {
            if (mSoundVersion != SoundOff && mMediaPlayers != null && sound != null)
            {
                // May be a crash culprit on some devices.  Wrapping in a try-catch.
                try
                {
                    MediaPlayer player = mMediaPlayers.GetNext();
                    player.Reset();
                    player.SetDataSource(sound);
                    player.Prepare();
                    player.Start();
                }
                catch { }
            }
        }

        public void HitDormantMine()
        {
            StartSingleSoundEffect(mDataHitDormantMine);
        }

        public void OrbCapture()
        {
            StartSingleSoundEffect(mDataOrbCapture);
        }

        public void MenuItem()
        {
            StartSingleSoundEffect(mDataMenuItem);
        }

        public void RoundComplete()
        {
            StartSingleSoundEffect(mDataRoundComplete);
        }

        public void RoundCompleteWin()
        {
            StartSingleSoundEffect(mDataRoundCompleteWin);
        }

        public void RoundEndEarly()
        {
            StartSingleSoundEffect(mDataRoundEndEarly);
        }
    }

    // This class is used to have multiple MediaPlayers per sound effect, so they don't always cancel the previous sound effect.
    public class MediaPlayerCycle
    {
        List<MediaPlayer> players;
        int index = 0;

        public MediaPlayerCycle(int count)
        {
            players = new List<MediaPlayer>();
            for (int j = 0; j < count; j++)
            {
                players.Add(new MediaPlayer());
            }
        }

        public MediaPlayer GetNext()
        {
            index++;
            if (index >= players.Count) { index = 0; }
            return players[index];
        }
    }

    public class SoundDataSource : MediaDataSource
    {
        private byte[] _data;

        public SoundDataSource(byte[] data)
        {
            _data = data;
        }

        public override long Size => _data.Length;

        public override void Close()
        {
            //_data = null; // Don't delete it because we will keep re-using this SoundDataSource
        }

        public override int ReadAt(long position, byte[] buffer, int offset, int size)
        {
            if (position >= _data.Length)
            {
                return -1;
            }

            if (position + size > _data.Length)
            {
                size -= (Convert.ToInt32(position) + size) - _data.Length;
            }
            Array.Copy(_data, position, buffer, offset, size);
            return size;
        }
    }
}