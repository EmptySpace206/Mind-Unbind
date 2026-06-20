using Android.App;
using Android.Content;
using Android.OS;
using Android.Runtime;
using Android.Views;
using Android.Widget;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MindMove.Cocos
{
    internal class Pulsate
    {
        const float PulsateScaleRange = SharedParams.OneThird;
        const float MaxPulsateScaleFactor = 1 + (PulsateScaleRange * SharedParams.TwoThirds);
        const float MinPulsateScaleFactor = 1 - (PulsateScaleRange * SharedParams.OneThird);
        const float PulsateRangeTotalTicks = 2718 * TimeSpan.TicksPerMillisecond; // 'e' seconds to pulsate
        const float MaxScaleFactorDelta = 0.05f;

        float mPulsateScaleFactor = 1;
        bool mPulsateDimming = false;

        long mLastPulsateTicks;

        public Pulsate()
        {
            mLastPulsateTicks = DateTime.UtcNow.Ticks;
        }

        public void NextScaleFactor()
        {
            long now = DateTime.UtcNow.Ticks;

            float delta = (float)(now - mLastPulsateTicks) * PulsateScaleRange / PulsateRangeTotalTicks;
            if (delta > MaxScaleFactorDelta) { delta = MaxScaleFactorDelta; }

            if (mPulsateDimming)
            {
                mPulsateScaleFactor -= delta;
                if (mPulsateScaleFactor <= MinPulsateScaleFactor) { mPulsateDimming = false; }
            }
            else
            {
                mPulsateScaleFactor += delta;
                if (mPulsateScaleFactor >= MaxPulsateScaleFactor) { mPulsateDimming = true; }
            }

            mLastPulsateTicks = now;
        }

        public void BurstScaleFactor()
        {
            mPulsateScaleFactor += (PulsateScaleRange / 2);
            if (mPulsateScaleFactor >= MaxPulsateScaleFactor) { mPulsateDimming = true; }
            if (mPulsateScaleFactor > MaxPulsateScaleFactor * (1+(PulsateScaleRange * 2))) { mPulsateScaleFactor = MaxPulsateScaleFactor * (1 + (PulsateScaleRange * 2)); }
        }

        public float GetScaleFactor()
        {
            return mPulsateScaleFactor;
        }

        public float GetReducedScaleFactor(float fraction)
        {
            return 1 + ((mPulsateScaleFactor - 1) * fraction);
        }

        public void ResetScaleFactor()
        {
            mPulsateScaleFactor = 1;
        }
    }
}