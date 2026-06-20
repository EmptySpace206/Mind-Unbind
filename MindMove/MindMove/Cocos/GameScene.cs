using CocosSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;

namespace MindMove.Cocos
{
    class GameScene : CCScene
    {
        MovePredictionEngineExt mEngine = null;
        ShortTouchHistory mHistory;
        MoveSimulator mSimulator;

        CCLayer mLayer;
        readonly float mSF;

        double mPositiveTotal = 0;
        double mNegativeTotal = 0;

        double mLastScore = 0;

        double mFinalScore = 0;
        double mLastFinalScore = 0;
        double mLastFinalScoreBonus = 0;
        int mMovesRemaining = 0;

        bool mReadyForDrawing = false;

        const int MinMovesToTriggerEnd = 1;

        double mUpdateDistance;
        double mUpdateDistanceDrawGranularity = SharedParams.DefaultDistancePerUpdate;
        bool mFixedGridGranularity = false;

        bool mSnapGridMode = false;
        bool mNoVirtualization = false;
        DrawLabel mSnapGridButton = null;
        DrawLabel mPureModeButton = null;
        bool mShowGridSnapMessage = false;

        int mMovesPerRound;
        const int DefaultMovesPerRound = 300;
        int[] TutorialMovesPerRoundCycle = { 80, 50, 30, 450, 300, 120 };
        int mTutorialMovesPerRoundIndex = 0;
        const int VoidStormMovesPerRond = 300;
        const int LongGameMovesPerRound = 450;
        const int HighlightHabitsMoveSequence = 30;
        const int ShortHighlightHabitsMoveSequence = 15;
        const int MovesPerRound_Continuity = 320;
        const int SeededMoves_Continuity = 40;
        const int MovesPerRound_Continuity_Long = 385;
        const int SeededMoves_Continuity_Long = 55;
        const int VoidStormMovesPerRoundGain = 30;

        const double HistoryScrollPercent = 0.1;
        const double ScoringMoveSequenceSearchStart = 0.25;
        const double MSPerMoveTimeLimit = 300;
        const double MSPerMoveTimeLimitSeeded = MSPerMoveTimeLimit * 2;
        const double MSPerMoveTimeLimitLongGame = MSPerMoveTimeLimit / ((double)LongGameMovesPerRound / (double)DefaultMovesPerRound);
        long mTimeRemainingTicks;
        long mRoundStartTicks;

        const int NumScoresToAverage = 20;
        List<double> mRecentScores;
        double mRecentAverageScore = 0;

        long mLastTouchMoveTicks;
        const float MediumFontSizeBase = 0.042f;
        readonly float MediumFontSize;
        const float LargeFontSizeBase = 0.048f;
        readonly float LargeFontSize;
        DrawLabel mScoreLabel = null;
        DrawLabel mControlLabel = null;
        DrawLabel mExtraInfoLabel = null;
        DrawLabel mExtraInfoLabel2 = null;

        DrawNode mGuide;
        DrawNode mGuideRing;
        DrawNode mTraceLine;
        DrawNode mScoreCircle;
        DrawLabel mDrawDistanceLabel;

        float GuideRadius;
        const float GuideLargeDotRadiusMult = 1 / 7.5f;
        const float GuideLargeDotRadiusMultGridSnap = 1 / 5f;
        float GuideLargeDotRadius;

        DrawColor mGuideBaseColor;

        const double MinScoreForTutorialFinish = 75;
        const double MinScoreForLeaderboard = 70;
        bool mReachedTutorialFinishScore = false;
        const double MinParticipationScore = 50;
        bool mReachedParticipationScore = false;

        int mNumberOfFinishedPlaysThisSession = 0;

        bool mEndRoundEarly = false;

        readonly MineManager mMineManager;
        const int StandardMineRespawnMoves = 30;
        int mStandardMovesSinceRespawn = 0;
        const int StandardMineInitialSpawnMoves = 10;
        const int StandardMineInitialSpawnMovesLongGame = 6;
        const int StandardMineCount = 3;
        const int StandardMineCountLongGame = 5;

        bool mResetMineCenters = true;
        long mTicksLastEmergeDecayDraw = 0;
        const int EmergDecayDrawPeriod = 33;

        int mConseqEndRoundEarlyDueToMines = 0; // For "don't touch mines, that's not the game!" tip

        ///////////////////////////////////////////////////////////////////////////////////////////
        // For Void Storm
        const double VoidStormMineSpawnRateIncreasePerLevel = 0.3;
        const double VoidStormScoreAdjustPerLevel = 0.1;
        double mVoidStormLevelScoreAdjuster = 0;
        const double VoidStormScoreBonusDecrementPerHit = 0.05;
        const double VoidStormBaseMineAreaIncrease = 1;

        const long VoidSpawnRateTicksBase = 222 * TimeSpan.TicksPerMillisecond;
        long mVoidSpawnRateTicks;
        long mLastMineSpawnTicks;
        const double VoidRemoveRateTicksMult = 1.5;
        long mVoidRemoveRateTicks;
        long mLastMineRemoveTicks;
        int mVoidStormLevel;
        const int MaxVoidStormLevel = 99;

        // End Void Storm
        ///////////////////////////////////////////////////////////////////////////////////////////

        // For Score Circle
        const float ScoreCirclePaddingMult = SharedParams.OneThird;
        const float ScoreCircleRadiusBase = 1 / 8f;
        float ScoreCircleRadius;
        int ScoreCircleTimerDotCount;
        const float ScoreCircleTimerDotRadiusMult = 0.075f;
        float ScoreCircleTimerDotRadius;
        const float ScoreFontSizeBase = 0.5833f;
        float ScoreFontSize;
        DrawPoint mScoreCircleTop;
        bool mScoreFeedbackHidden = false;

        List<GlitterDot> mGlitterDots;
        bool mDrawingGlitter = false;
        float DefaultGlitterRadiius;
        readonly float MinGlitterRadius = 2.0f;

        const float TraceLineWidthBaseOfScoreCircle = 0.02f;
        float TraceLineWidth;
        const float MinTraceLineWidth = 1.0f;

        List<GlitterDot> mIdleGlitterDots;
        const int NumIdleGlitterDots = 45;
        const int NumIdleGlitterDotsMuted = 22;
        const float IdleGlitterMutedRadiusMult = 0.75f;
        const float GlitterEffect100PlusBoost = 1.2f;

        List<ScoredDraw> mScoredDraws; 

        ///////////////////////////////////////////////////////////////////////////////////////////
        // For Configurations ("continuity")
        int mContinuityScoredDrawStartIndex = 0;
        int mContinuityScoredDrawEndIndex = 0;
        bool mInitializedContinuityScoredDrawStartIndex = false;
        bool mIsContinuityStarted = false;
        int mContinuityScoredDrawFirstMoveThisGame = 0;
        const int ContinuityGamesCount = 3;
        const int ContinuityGamesCount_Long = 4;
        readonly List<double> mContinuitySeriesScores = null;
        double mContinuityLastSeriesScore = 0;
        double mContinuitySeriesBestScore = 0;
        double mContinuityLongSeriesBestScore = 0;
        bool mDelayContinuityGameStart = false;

        bool mLongContinuitySeries = false;
        int mContinuityGamesCount = ContinuityGamesCount;
        int mMovesPerRoundContinuity = MovesPerRound_Continuity;
        int mSeededMovesContinuity = SeededMoves_Continuity;
        // End Configurations ("continuity")
        ///////////////////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////////////////////
        // For Conequitive games at 100+ 
        const string ConseqGames100Plus = "ConseqGames100Plus.xml";
        const string SumScores100Plus = "SumScores100Plus.xml";
        const string BestConseqGames100Plus = "BestConseqGames100Plus.xml";
        int mConseqGamesScoring100Plus = 0;
        double mSumScores100Plus = 0;
        int mBestConseqGamesScoring100Plus = 0;
        // End Conequitive games
        ///////////////////////////////////////////////////////////////////////////////////////////

        bool mShowingSpecialDetails = false;

        ThemeColors mThemeColors;

        public enum PlayMode
        {
            Standard,
            VoidStormV2,
            Continuity
        }
        readonly PlayMode mPlayMode;

        // For intro text
        bool mFirstPlay = true;
        DrawLabel mHeaderText;

        // For best and historical scores
        double mBestScore;
        double mVoidStormBestScore = 0;
        double mBestLongGameScore = 0;
        const string ReachedNormalScoreCircleSizeThreshold = "ReachedNormalScoreCircleSizeThresholdV1.xml";
        const string ReachedParticipationThreshold = "ReachedParticipationThresholdV1.xml";
        const string TargetAnglesEnabled = "TargetAnglesEnabled.xml";
        const string LongSeriesEnabled = "LongSeriesEnabled.xml";
        const string VoidStormOrbLevel = "VoidStormOrbLevel.xml";
        const string CurrentUpdateGranularity = "CurrentUpdateGranularity.xml";
        const string FixedUpdateGranularity = "FixedUpdateGranularity.xml";
        const string SnapToGridMode = "SnapToGridMode.xml";

        double mSumDegrees;

        bool mLongGameEnabled = false;

        DrawNode mGameModeCircle;
        DrawLabel mGameModeLabel;
        DrawPoint mGameModeCircleCenter;
        const float GameModeCircleRadiusFractionOfScoreCircle = 0.533f;
        float GameModeCircleRadius;

        public const double StateNotInit = -1;
        double mLatestRelativeAngle;
        double mLastState = StateNotInit;
        double mLastState2 = StateNotInit;
        double mLastStateVirt = StateNotInit;

        DrawLabel mTimerOrBonusLabel;
        DrawColor mTimerOrBonusLabelColorBase;
        const double TimerColorBaseMult = 0.5;

        // For pulsate
        long mTicksLastPulsate = 0;
        readonly Pulsate mPulsate;

        DrawLabel mRankLabel;
        DrawLabel mBestAndNextRankLabel;
        bool mRedrawRankLabel = false;

        // For draw area bounds
        DrawRect mDrawAreaBounds;
        int mNumOutOfBoundsMoves = 0;
        const int OutOfBoundsPenaltyMoves = 5;
        DrawNode mBoundsLine;
        DrawColor mBoundsLineColor;
        readonly float DefaultBoundsWidth = 2;

        // For details overlay
        bool mExperimentMode = false;
        DrawNode mDetailsOverlay;
        readonly List<DrawLabel> mDetailsTexts;
        readonly List<RankBadge> mDetailsBadges;
        DrawLabel mExperimentModeToggle;

        // For text underlays
        DrawNode mTextUnderlay;

        // For badge
        RankBadge mRankBadge;
        float mRankBadgeRadius = 0;
        DrawPoint mRankBadgeCenter;

        // For border background gradient
        DrawNode mGridAndBorder;

        // For color picker
        DrawNode mColorPicker;

        readonly Random mRnd;

        // Locks
        bool mUpdateThreadLocked = true;

        public GameScene(CCGameView gameView, string playMode, ThemeColors themeColors) : base(gameView)
        {
            mSF = MainActivity.Current.ScalingFactor();

            mLayer = new CCLayer();
            this.AddLayer(mLayer);

            mRnd = new Random();

            MinGlitterRadius *= mSF;
            DefaultBoundsWidth *= mSF;

            MediumFontSize = mLayer.ContentSize.Width * MediumFontSizeBase;
            LargeFontSize = mLayer.ContentSize.Width * LargeFontSizeBase;
            if (mLayer.ContentSize.Height / mLayer.ContentSize.Width < SharedParams.SizeAdjustRatioThreshold)
            {
                float mult = (mLayer.ContentSize.Height / mLayer.ContentSize.Width) / SharedParams.SizeAdjustRatioThreshold;
                MediumFontSize *= mult;
                LargeFontSize *= mult;
            }

            if (playMode == SharedParams.VoidStorm)
            {
                mPlayMode = PlayMode.VoidStormV2;
            }
            else if (playMode == SharedParams.StreamOfContinuity)
            {
                mPlayMode = PlayMode.Continuity;
                mContinuitySeriesScores = new List<double>();
                mLongContinuitySeries = MainActivity.GetSavedIntWithDefaultValue(0, LongSeriesEnabled) != 0;
            }
            else
            {
                mPlayMode = PlayMode.Standard;
            }

            SetScoreCircleRadius();

            mGlitterDots = new List<GlitterDot>();
            mIdleGlitterDots = new List<GlitterDot>();

            mScoredDraws = new List<ScoredDraw>();

            mDetailsTexts = new List<DrawLabel>();
            mDetailsBadges = new List<RankBadge>();

            mRecentScores = new List<double>();

            mThemeColors = themeColors;

            mPulsate = new Pulsate();


            mReachedTutorialFinishScore = (MainActivity.GetSavedIntWithDefaultValue(0, ReachedNormalScoreCircleSizeThreshold) != 0);
            mReachedParticipationScore = (MainActivity.GetSavedIntWithDefaultValue(0, ReachedParticipationThreshold) != 0);

            mLongGameEnabled = (MainActivity.GetSavedIntWithDefaultValue(0, TargetAnglesEnabled) != 0);
            if (mPlayMode != PlayMode.Standard) { mLongGameEnabled = false; }   

            mFixedGridGranularity = (MainActivity.GetSavedIntWithDefaultValue(0, FixedUpdateGranularity) != 0);

            if (MainActivity.GetSavedIntWithDefaultValue(0, SnapToGridMode) != 0)
            {
                mSnapGridMode = true;
            }
            else
            {
                mSnapGridMode = false;
            }

            if (mPlayMode == PlayMode.VoidStormV2)
            {
                mVoidStormLevel = MainActivity.GetSavedIntWithDefaultValue(0, VoidStormOrbLevel);
                if (mVoidStormLevel > MaxVoidStormLevel) { mVoidStormLevel = MaxVoidStormLevel; }
                mSnapGridMode = false;
            }

            mMineManager = new MineManager(mLayer, mThemeColors, mPlayMode);
        }

        public override void OnEnterTransitionDidFinish()
        {
            base.OnEnterTransitionDidFinish();

            CreateTouchListeners();

            StartNewRound();

            this.Schedule();
        }

        public override void OnExit()
        {
            base.OnExit();

            try
            {
                mLayer.RemoveAllChildren();
            }
            catch { }
        }

        void CreateTouchListeners()
        {
            var touchListener = new CCEventListenerTouchAllAtOnce();
            touchListener.OnTouchesBegan = HandleTouchesBegan;
            touchListener.OnTouchesMoved = HandleTouchesMoved;
            touchListener.OnTouchesEnded = HandleTouchesEnded;
            touchListener.OnTouchesCancelled = HandleTouchesEnded;
            mLayer.AddEventListener(touchListener);
        }

        void StartNewRound()
        {
            mUpdateThreadLocked = true;
            mReadyForDrawing = false;
            mPulsate.ResetScaleFactor();

            mLayer.RemoveAllChildren();

            if (mPlayMode == PlayMode.Continuity)
            {
                if (mLongContinuitySeries)
                {
                    mMovesPerRoundContinuity = MovesPerRound_Continuity_Long;
                    mSeededMovesContinuity = SeededMoves_Continuity_Long;
                    mContinuityGamesCount = ContinuityGamesCount_Long;
                }
                else
                {
                    mMovesPerRoundContinuity = MovesPerRound_Continuity;
                    mSeededMovesContinuity = SeededMoves_Continuity;
                    mContinuityGamesCount = ContinuityGamesCount;
                }
            }

            GetBestScores();

            mThemeColors.CreateBackground(ref mLayer);
            mGuideBaseColor = ThemeColors.AverageColors(mThemeColors.DistinctColorA, mThemeColors.DistinctColorB);

            mTimerOrBonusLabelColorBase = mPlayMode == PlayMode.VoidStormV2 ? mThemeColors.Text : ThemeColors.AdjustColorsForMultipler(TimerColorBaseMult, mThemeColors.Text);

            SetDrawAreaBounds();
            DrawGridAndBorder();

            mBoundsLine = new DrawNode();
            mBoundsLineColor = ThemeColors.GetColorForTargetLuminance(mThemeColors.Color3, 0.475, false);
            mLayer.AddChild(mBoundsLine);
            DrawBoundsLine(DefaultBoundsWidth);

            if (IsInContinuitySeries())
            {
                if (!mInitializedContinuityScoredDrawStartIndex)
                {
                    GetMostExtremeScoringIndexes(mSeededMovesContinuity, out mContinuityScoredDrawStartIndex);
                    mContinuityScoredDrawEndIndex = mContinuityScoredDrawStartIndex + mSeededMovesContinuity;
                    mRoundStartTicks = DateTime.UtcNow.Ticks;
                }

                ReAddAllLinesToLayer(mContinuityScoredDrawStartIndex, mContinuityScoredDrawEndIndex);
            }
            else // !IsInContinuitySeries()
            { 
                mLastState = mLastState2 = StateNotInit;
                mLatestRelativeAngle = 90;
            }

            mHistory = new ShortTouchHistory(mSnapGridMode);

            if (mPlayMode != PlayMode.Continuity || !mInitializedContinuityScoredDrawStartIndex)
            {
                SetPlayModeParameters();
            }

            mLastFinalScore = mFinalScore;
            mFinalScore = 0;
            mLastScore = 0;
            mRecentAverageScore = 0;

            mRecentScores.Clear();
            for (int j = 0; j < NumScoresToAverage; j++) { mRecentScores.Add(0); }

            UpdateScoreCircle(true);

            mMovesRemaining = mMovesPerRound;
            mNumOutOfBoundsMoves = 0;

            mLastMineSpawnTicks = 0;
            mLastMineRemoveTicks = 0;

            if (mResetMineCenters)
            {
                double drawArea = Math.Sqrt(mDrawAreaBounds.Size.Width * mDrawAreaBounds.Size.Height);
                mMineManager.InitializeMines(mDrawAreaBounds, drawArea, 1, mPlayMode != PlayMode.VoidStormV2, mLongGameEnabled);
            }
            mMineManager.SetThemeColor(mThemeColors, mThemeColors.DualColorGradient);
            mMineManager.DrawStartMines(true);

            if (!mEndRoundEarly) { mConseqEndRoundEarlyDueToMines = 0; }
            mEndRoundEarly = false;

            mNegativeTotal = 0;
            mPositiveTotal = 0;

            mDrawHistoryVisual = null;
            mDrawHistoryVisualColoredLine = null;

            if (mPlayMode == PlayMode.VoidStormV2) { mVoidStormLevelScoreAdjuster = VoidStormScoreAdjustPerLevel * mVoidStormLevel; }
            else { mVoidStormLevelScoreAdjuster = 0; }

            mSumDegrees = 0;
            if (mPlayMode == PlayMode.Standard && mBestScore >= SharedParams.RoBMUnlockBaseScore)
            {
                UpdateGameModeCircle(true);
            }
            else if (mPlayMode == PlayMode.Continuity && mContinuitySeriesBestScore >= SharedParams.ContinuityLongSeriesUnlock
                && !IsInContinuitySeries())
            {
                UpdateGameModeCircle(true);
            }
            else if (mPlayMode == PlayMode.VoidStormV2 && mVoidStormBestScore >= SharedParams.StormLevel1Score)
            {
                UpdateGameModeCircle(true);
            }

            CreateLabels();
            if (mFirstPlay)
            {
                DrawRankLabel();
            }

            mTraceLine = new DrawNode();
            mTraceLine.IsOpacityCascaded = true;
            mTraceLine.IsColorModifiedByOpacity = true;
            mLayer.AddChild(mTraceLine);

            mGuide = new DrawNode();
            mLayer.AddChild(mGuide);

            mGuideRing = new DrawNode();
            mLayer.AddChild(mGuideRing);

            if (mGlitterDots.Count > 0)
            {
                mDrawingGlitter = true;
            }

            ReAddAllIdleGlitterToLayer();

            DrawColorPicker();

            if (IsInContinuitySeries())
            {
                MinimizeScreenContent();

                RedrawAllLines(mContinuityScoredDrawStartIndex, false, mContinuityScoredDrawEndIndex);

                // BUG: After coming out of Show Last Draw, this draws artifacts? It's because of highlight lines
                int endIndex = mContinuityScoredDrawEndIndex > 0 ? mContinuityScoredDrawEndIndex - 1 : mScoredDraws.Count - 1;
                mScoredDraws[endIndex].DrawStartEndIndicatorArrow(mThemeColors.Text, TraceLineWidth, (float)mUpdateDistance);
            }
            else if (mContinuitySeriesScores != null)
            {
                mContinuitySeriesScores.Clear();
            }

            mInitializedContinuityScoredDrawStartIndex = true;

            mUpdateThreadLocked = false;
        }

        void SeedGame()
        {
            const int UpdateTicks = 5;
            mEngine = null;
            mSimulator = null;
            mScoredDraws.Clear();

            mEngine = new MovePredictionEngineExt(mMovesPerRound + mSeededMovesContinuity);
            mSimulator = new MoveSimulator(mDrawAreaBounds, (float)mUpdateDistance, TraceLineWidth, useVirtualization: !mNoVirtualization);

            int moves = 0;

            double distance = mSnapGridMode ? mUpdateDistance * 1.05 : mUpdateDistance / UpdateTicks;
            while (moves < mSeededMovesContinuity)
            {
                if (mSimulator.SimMove(
                    MoveSimulator.SimMode.Targeted,
                    distance,
                    ref mLayer,
                    ref mHistory,
                    ref mEngine,
                    mThemeColors))
                {
                    moves++;
                }
            }

            mScoredDraws.AddRange(mSimulator.GetScoredDraws());

            mSimulator.GetRecentVirtalizedMoves(out mLastState, out mLastState2, out mLastStateVirt, out mLatestRelativeAngle);

            mHistory.Clear();

            mContinuityLastSeriesScore = 0;
        }

        void DrawGridAndBorder(bool init = true, bool pulsate = false)
        {
            if (init)
            {
                if (mGridAndBorder != null)
                {
                    mGridAndBorder.Dispose();
                    mLayer.RemoveChild(mGridAndBorder);
                    mGridAndBorder = null;
                }

                mGridAndBorder = new DrawNode();
                mLayer.AddChild(mGridAndBorder);
            }
            else if (mGridAndBorder == null) { return; }
            else
            {
                mGridAndBorder.Clear();
                mGridAndBorder.Cleanup();
                mGridAndBorder.Visible = false;
            }

            // Note: We assume that the top and bottom widths are equal
            float height = mDrawAreaBounds.MinY;

            // Draw a ligher top and bottom
            DrawColor lightBkg1 = ThemeColors.AdjustColorsForMultipler(2.0, mThemeColors.BackgroundColor);
            DrawColor lightBkg2 = ThemeColors.AdjustColorsForMultipler(1.75, mThemeColors.BackgroundColor);
            mGridAndBorder.DrawLine(
                from: new DrawPoint(-height, 0),
                to: new DrawPoint(mLayer.ContentSize.Width + height, 0),
                color: lightBkg1,
                lineWidth: height);
            mGridAndBorder.DrawLine(
                from: new DrawPoint(-height, mLayer.ContentSize.Height),
                to: new DrawPoint(mLayer.ContentSize.Width + height, mLayer.ContentSize.Height),
                color: lightBkg1,
                lineWidth: height);

            // Draw dots indicating update distance grid
            for (double j = mLayer.ContentSize.Center.X; j >= 0; j -= mUpdateDistance)
            {
                for (double i = mLayer.ContentSize.Center.Y; i >= 0; i -= mUpdateDistance)
                {
                    mGridAndBorder.DrawSolidCircle(new DrawPoint((float)j, (float)i),
                        i > mDrawAreaBounds.MinY ? (float)mUpdateDistance / 13.33f : (float)mUpdateDistance / 9f,
                        i > mDrawAreaBounds.MinY ? lightBkg1 : lightBkg2);
                }

                for (double i = mLayer.ContentSize.Center.Y; i <= mLayer.ContentSize.Height; i += mUpdateDistance)
                {
                    mGridAndBorder.DrawSolidCircle(new DrawPoint((float)j, (float)i),
                        i < mDrawAreaBounds.MaxY ? (float)mUpdateDistance / 13.33f : (float)mUpdateDistance / 9f,
                        i < mDrawAreaBounds.MaxY ? lightBkg1 : lightBkg2);
                }
            }

            for (double j = mLayer.ContentSize.Center.X; j <= mLayer.ContentSize.Width; j += mUpdateDistance)
            {
                for (double i = mLayer.ContentSize.Center.Y; i <= mLayer.ContentSize.Height; i += mUpdateDistance)
                {
                    mGridAndBorder.DrawSolidCircle(new DrawPoint((float)j, (float)i),
                        i < mDrawAreaBounds.MaxY ? (float)mUpdateDistance / 13.33f : (float)mUpdateDistance / 9f,
                        i < mDrawAreaBounds.MaxY ? lightBkg1 : lightBkg2);
                }

                for (double i = mLayer.ContentSize.Center.Y; i >= 0; i -= mUpdateDistance)
                {
                    mGridAndBorder.DrawSolidCircle(new DrawPoint((float)j, (float)i),
                        i > mDrawAreaBounds.MinY ? (float)mUpdateDistance / 13.33f : (float)mUpdateDistance / 9f,
                        i > mDrawAreaBounds.MinY ? lightBkg1 : lightBkg2);
                }
            }

            mGridAndBorder.Visible = true;
        }

        void DrawColorPicker(bool pulsate = false)
        {
            DrawColor color1 = mThemeColors.Color1;
            DrawColor color2 = mThemeColors.Color2;
            DrawColor color3 = mThemeColors.Color3;

            if (!pulsate)
            {
                mColorPicker = new DrawNode();
                mLayer.AddChild(mColorPicker);
            }
            else
            {
                mColorPicker.Clear();
                mColorPicker.Cleanup();
                mColorPicker.Visible = false;

                // TODO: Set basic pulsating colors globally, instead of in each local function
                color1 = ThemeColors.AdjustColorsForMultipler(mPulsate.GetScaleFactor(), color1.R, color1.G, color1.B);
                color2 = ThemeColors.AdjustColorsForMultipler(mPulsate.GetScaleFactor(), color2.R, color2.G, color2.B);
                color3 = ThemeColors.AdjustColorsForMultipler(mPulsate.GetScaleFactor(), color3.R, color3.G, color3.B);
            }

            float radius = ScoreCircleRadius / 4.667f;
            float outerRadius = radius * 1.1667f;
            float x = mLayer.ContentSize.Width * 0.9f;
            float y = mDrawAreaBounds.MinY * 0.5f;

            mColorPicker.DrawSolidCircle(
                new DrawPoint(x - radius, y),
                outerRadius,
                mThemeColors.AveragedColor12);
            mColorPicker.DrawSolidCircle(
                new DrawPoint(x, y),
                outerRadius,
                mThemeColors.AveragedColor23);
            mColorPicker.DrawSolidCircle(
                new DrawPoint(x + radius, y),
                outerRadius,
                mThemeColors.AveragedColor13);
            mColorPicker.DrawSolidCircle(
                new DrawPoint(x - radius, y),
                radius,
                color3);
            mColorPicker.DrawSolidCircle(
                new DrawPoint(x, y),
                radius,
                color1);
            mColorPicker.DrawSolidCircle(
                new DrawPoint(x + radius, y),
                radius,
                color2);

            mColorPicker.Visible = true;
        }

        void SetScoreCircleRadius(double radMult = 1)
        {
            ScoreCircleRadius = mLayer.ContentSize.Width * ScoreCircleRadiusBase * (float)radMult;
            if (mLayer.ContentSize.Height / mLayer.ContentSize.Width < SharedParams.SizeAdjustRatioThreshold)
            {
                float mult = (mLayer.ContentSize.Height / mLayer.ContentSize.Width) / SharedParams.SizeAdjustRatioThreshold;
                ScoreCircleRadius *= mult;
            }

            ScoreFontSize = ScoreFontSizeBase * ScoreCircleRadius;
            GameModeCircleRadius = ScoreCircleRadius * GameModeCircleRadiusFractionOfScoreCircle;

            ScoreCircleTimerDotRadius = ScoreCircleRadius * ScoreCircleTimerDotRadiusMult;

            TraceLineWidth = ScoreCircleRadius * TraceLineWidthBaseOfScoreCircle;
            if (TraceLineWidth < MinTraceLineWidth) { TraceLineWidth = MinTraceLineWidth; }

            DefaultGlitterRadiius = (float)(ScoreCircleRadius * 0.055);
        }

        void SetDrawAreaBounds()
        {
            float scoreCircleRadius = ScoreCircleRadius;

            float padding = ScoreCirclePaddingMult;

            mDrawAreaBounds = new DrawRect(
                0,
                0 + scoreCircleRadius * (1 + padding) * 2,
                mLayer.ContentSize.Width,
                mLayer.ContentSize.Height - scoreCircleRadius * (1 + padding) * 4);

            SetUpdateDistance();
        }

        // Set the update distance, and all related parameters
        void SetUpdateDistance()
        {
            double drawArea = Math.Sqrt(mDrawAreaBounds.Size.Width * mDrawAreaBounds.Size.Height);

            if (mSnapGridMode)
            {
                mUpdateDistanceDrawGranularity = SharedParams.UpdateDistanceSnapGrid;
            }
            else if (mFixedGridGranularity)
            {
                mUpdateDistanceDrawGranularity = SharedParams.MinDistancePerUpdate;
            }
            else
            {
                mUpdateDistanceDrawGranularity = MainActivity.GetSavedIntWithDefaultValue(0, CurrentUpdateGranularity) / 10000.0;
                if (mUpdateDistanceDrawGranularity == 0) { mUpdateDistanceDrawGranularity = SharedParams.DefaultDistancePerUpdate; }
                else if (mUpdateDistanceDrawGranularity > SharedParams.MaxDistancePerUpdate) { mUpdateDistanceDrawGranularity = SharedParams.MaxDistancePerUpdate; }
                else if (mUpdateDistanceDrawGranularity < SharedParams.MinDistancePerUpdate) { mUpdateDistanceDrawGranularity = SharedParams.MinDistancePerUpdate; }
            }

            mUpdateDistance = mUpdateDistanceDrawGranularity * drawArea;

            SetGuideParameters();
        }

        void SetGuideParameters()
        {
            if (!mSnapGridMode) 
            {
                GuideRadius = (float)(mUpdateDistance * 1.2f);
                GuideLargeDotRadius = GuideRadius * GuideLargeDotRadiusMult;
            } 
            else
            {
                // Make the radius the update distance, but slightly enlarge the dots. 
                GuideRadius = (float)mUpdateDistance;
                GuideLargeDotRadius = GuideRadius * GuideLargeDotRadiusMultGridSnap;
                GuideRadius += GuideLargeDotRadius;
            }
        }

        const float IncreasedScaleFactorPulsingUI = 0.125f;
        void DrawBoundsLine(float width, double colorMult = 1, bool pulsateMode = false, DrawColor? colorOverride = null)
        {
            mBoundsLine.Clear();
            mBoundsLine.Cleanup();
            mBoundsLine.Visible = false;

            DrawColor color;
            if (pulsateMode)
            {
                color = ThemeColors.AdjustColorsForMultipler(mPulsate.GetReducedScaleFactor(1 + IncreasedScaleFactorPulsingUI) * colorMult, mBoundsLineColor);
            }
            else if (colorOverride != null)
            {
                color = colorOverride.Value;
            }
            else
            {
                color = ThemeColors.AdjustColorsForMultipler(colorMult, mBoundsLineColor);
            }

            mBoundsLine.DrawLine(
                from: new DrawPoint(mDrawAreaBounds.MinX, mDrawAreaBounds.MinY),
                to: new DrawPoint(mDrawAreaBounds.MaxX, mDrawAreaBounds.MinY),
                color: color,
                lineWidth: width);
            mBoundsLine.DrawLine(
                from: new DrawPoint(mDrawAreaBounds.MinX, mDrawAreaBounds.MaxY),
                to: new DrawPoint(mDrawAreaBounds.MaxX, mDrawAreaBounds.MaxY),
                color: color,
                lineWidth: width);

            mBoundsLine.Visible = true;
        }

        void SetPlayModeParameters()
        {
            if (mPlayMode == PlayMode.VoidStormV2)
            {
                mMovesPerRound = VoidStormMovesPerRond;

                mMovesPerRound += mVoidStormLevel * VoidStormMovesPerRoundGain;
            }
            else if (mPlayMode == PlayMode.Continuity)
            {
                mMovesPerRound = mMovesPerRoundContinuity - mSeededMovesContinuity;
            }
            else
            {
                if (mExperimentMode)
                {
                    mMovesPerRound = TutorialMovesPerRoundCycle[mTutorialMovesPerRoundIndex];
                }
                else if (mLongGameEnabled)
                {
                    mMovesPerRound = LongGameMovesPerRound;
                }
                else
                {
                    mMovesPerRound = DefaultMovesPerRound;
                }
            }

            mVoidSpawnRateTicks = (long)(VoidSpawnRateTicksBase / (1.0 + (VoidStormMineSpawnRateIncreasePerLevel * mVoidStormLevel)));

            mVoidRemoveRateTicks = (long)(mVoidSpawnRateTicks * VoidRemoveRateTicksMult);
        }

        void ReadyForDrawing()
        {
            if (mReadyForDrawing) { return; }

            mReadyForDrawing = true;

            if (mPlayMode != PlayMode.Continuity)
            {
                mEngine = null;
                mEngine = new MovePredictionEngineExt(mMovesPerRound);

                ClearAllLines(true);
                mScoredDraws.Clear();

                mContinuityScoredDrawFirstMoveThisGame = 0;

                mRoundStartTicks = DateTime.UtcNow.Ticks;
            }
            else if (!IsInContinuitySeries())
            {
                SeedGame();

                mContinuityScoredDrawFirstMoveThisGame = mScoredDraws.Count;
                mContinuityScoredDrawStartIndex = 0;
                mContinuityScoredDrawEndIndex = 0;
                mIsContinuityStarted = true;

                mScoredDraws[mScoredDraws.Count - 1].DrawStartEndIndicatorArrow(mThemeColors.Text, TraceLineWidth, (float)mUpdateDistance);
                ShowDrawHistory();

                mReadyForDrawing = false;

                mRoundStartTicks = DateTime.UtcNow.Ticks;
            }
            else if (mContinuitySeriesScores.Count > 0 && (mContinuityScoredDrawStartIndex > 0 || mContinuityScoredDrawEndIndex > 0) && mEngine != null)
            {
                mScoredDraws = mScoredDraws.GetRange(mContinuityScoredDrawStartIndex, (mContinuityScoredDrawEndIndex - mContinuityScoredDrawStartIndex));

                List<double> directions = mEngine.GetDegreesHistory().GetRange(mContinuityScoredDrawStartIndex, (mContinuityScoredDrawEndIndex - mContinuityScoredDrawStartIndex));
                List<double> tempDirStates = new List<double>();
                for(int j = 0; j < directions.Count; j++)
                {
                    tempDirStates.Add(directions[j]);
                }

                mContinuityScoredDrawStartIndex = 0;
                mContinuityScoredDrawEndIndex = 0;

                mContinuityScoredDrawFirstMoveThisGame = mScoredDraws.Count;

                mEngine = null;
                mEngine = new MovePredictionEngineExt(mMovesPerRound + mScoredDraws.Count);
                for (int j = 0; j < tempDirStates.Count; j++)
                {
                    mEngine.RecordMove(tempDirStates[j]);
                }
            }

            DrawControlLabel();
            mShowingSpecialDetails = false;

            mFirstPlay = false;

            CleanRankLabelContent();
            RemoveTextUnderlay();

            mLayer.RemoveChild(mHeaderText);
            mLayer.RemoveChild(mExtraInfoLabel);
            mLayer.RemoveChild(mExtraInfoLabel2);
            mLayer.RemoveChild(mColorPicker);

            mLineRedrawIndex = LineRedrawOff;

            if (mReadyForDrawing)
            {
                mMineManager.DrawStartMines(false);
                DrawBoundsLine(DefaultBoundsWidth);

                PauseIdleGlitterDrawing();

                if (mGameModeCircle != null)
                {
                    mGameModeCircle.Visible = false;
                }
                mLayer.RemoveChild(mGameModeLabel);
                mLayer.RemoveChild(mExperimentModeToggle);
                mLayer.RemoveChild(mSnapGridButton);
                mLayer.RemoveChild(mPureModeButton);

                mLastTouchMoveTicks = DateTime.UtcNow.Ticks;
                mLastMineSpawnTicks = mLastTouchMoveTicks;
                mLastMineRemoveTicks = mLastTouchMoveTicks;
                mStandardMovesSinceRespawn = 0;

                mLastFinalScoreBonus = 0;
            }
        }

        void CleanRankLabelContent()
        {
            mRedrawRankLabel = false;
            if (mRankBadge != null)
            {
                mRankBadge.Deactivate();
                mRankBadge = null;
            }

            mLayer.RemoveChild(mRankLabel);
            mLayer.RemoveChild(mBestAndNextRankLabel);
        }

        void MinimizeScreenContent(bool minimize = true, bool v2 = false)
        {
            mMineManager.HideMines(minimize);
            if (mGameModeLabel != null) { mGameModeLabel.Visible = !minimize; }
            if (mGameModeCircle != null) { mGameModeCircle.Visible = !minimize; }
            if (mColorPicker != null) { mColorPicker.Visible = !minimize; }
            if (mSnapGridButton != null) { mSnapGridButton.Visible = !minimize; }
            if (mPureModeButton != null) { mPureModeButton.Visible = !minimize; }
        }

        void UpdateScoreCircle(bool init, bool pulsateMode = false)
        {
            const double ColorBrightenerOffset = 0.2;
            DrawColor color2 = mThemeColors.Color2;
            DrawColor color3 = mThemeColors.AveragedColor23;

            float scoreCircleRadius = ScoreCircleRadius;
            float scoreCircleTimerDotRadius = ScoreCircleTimerDotRadius;

            float timerDotRadiusMult = 1.0f;
            if (!mReadyForDrawing)
            {
                if (mLastFinalScore > 1)
                {
                    timerDotRadiusMult *= (float)mLastFinalScore * GlitterEffect100PlusBoost;
                    scoreCircleTimerDotRadius *= timerDotRadiusMult;
                }
            }

            if (init)
            {
                float padding = ScoreCirclePaddingMult;

                ScoreCircleTimerDotCount = (int)(mMovesPerRound / 4);
                mScoreCircleTop = new DrawPoint(mLayer.ContentSize.Width * 0.5f, mLayer.ContentSize.Height - (scoreCircleRadius * (1 + padding)));

                // Draw the rings -- we only need to draw these once
                DrawNode rings = new DrawNode();
                mLayer.AddChild(rings);

                DrawColor brightColor3 = ThemeColors.AdjustColorsForMultipler(1 + ColorBrightenerOffset, color3.R, color3.G, color3.B);

                rings.DrawCircle(mScoreCircleTop, scoreCircleRadius, brightColor3);
                rings.DrawCircle(mScoreCircleTop, scoreCircleRadius + 1 * mSF, brightColor3);
                rings.DrawCircle(mScoreCircleTop, scoreCircleRadius - 1 * mSF, brightColor3);

                mScoreCircle = new DrawNode();
                mLayer.AddChild(mScoreCircle);

                mDrawDistanceLabel = new DrawLabel("" + mMovesRemaining, "Arial", ScoreFontSize * 0.5f, CCLabelFormat.SystemFont);
                mDrawDistanceLabel.Color = new CCColor3B(ThemeColors.AdjustColorsForMultipler(0.8, mThemeColors.TextColor2));
                mDrawDistanceLabel.PositionX = mScoreCircleTop.X;
                mDrawDistanceLabel.PositionY = mScoreCircleTop.Y + ScoreFontSize * 1.125f;
                mDrawDistanceLabel.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mDrawDistanceLabel);

                mTimerOrBonusLabel = new DrawLabel("", "Arial", ScoreFontSize * 0.5f, CCLabelFormat.SystemFont);
                mTimerOrBonusLabel.Color = new CCColor3B(mTimerOrBonusLabelColorBase);
                mTimerOrBonusLabel.PositionX = mScoreCircleTop.X;
                mTimerOrBonusLabel.PositionY = mScoreCircleTop.Y - ScoreFontSize;
                mTimerOrBonusLabel.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mTimerOrBonusLabel);
            }

            if (pulsateMode)
            {
                scoreCircleTimerDotRadius *= mPulsate.GetReducedScaleFactor(0.4f * timerDotRadiusMult);
                color2 = ThemeColors.AdjustColorsForMultipler(mPulsate.GetScaleFactor(), color2.R, color2.G, color2.B);
            }

            mScoreCircle.Clear();
            mScoreCircle.Cleanup();
            mScoreCircle.Visible = false;

            // Draw timer dots
            if (!mEndRoundEarly)
            {
                double radianStep = (2 * Math.PI) / (double)ScoreCircleTimerDotCount;
                int numDots = (int)Math.Ceiling(ScoreCircleTimerDotCount * ((double)mMovesRemaining / (double)mMovesPerRound));
                int offset = ScoreCircleTimerDotCount / 4;
                for (int j = 0 + offset; j < numDots + offset; j++)
                {
                    float x = mScoreCircleTop.X + (scoreCircleRadius * ((float)Math.Cos(((double)j * radianStep))));
                    float y = mScoreCircleTop.Y + (scoreCircleRadius * ((float)Math.Sin(((double)j * radianStep))));

                    mScoreCircle.DrawSolidCircle(new DrawPoint(x, y), scoreCircleTimerDotRadius, color2);
                }
            }

            // Update the draw distance
            mDrawDistanceLabel.Text = "" + mMovesRemaining;

            // Update the timer or bonus
            if (mPlayMode == PlayMode.VoidStormV2)
            {
                UpdateVoidStormTimerOrBonusLabel();
            }

            mScoreCircle.Visible = true;
        }

        void UpdateTimeRemaining()
        {
            if (IsInContinuitySeries() ||
                (mReadyForDrawing && !mExperimentMode && 
                (mMovesPerRound != mMovesRemaining && mMovesRemaining != 0) && 
                mPlayMode != PlayMode.VoidStormV2))
            {
                long maxTicks = (long)(mMovesPerRound * (mLongGameEnabled ? MSPerMoveTimeLimitLongGame : MSPerMoveTimeLimit) * TimeSpan.TicksPerMillisecond);
                if (IsInContinuitySeries()) { maxTicks += (long)(mSeededMovesContinuity * MSPerMoveTimeLimitSeeded * TimeSpan.TicksPerMillisecond); }
                mTimeRemainingTicks = maxTicks - (DateTime.UtcNow.Ticks - mRoundStartTicks);

                if (mTimeRemainingTicks <= 0)
                {
                    mTimeRemainingTicks = 0;
                    if (!mEndRoundEarly) { MainActivity.SoundEffects.RoundEndEarly(); }
                    mEndRoundEarly = true;
                }

                string text = Math.Round((double)mTimeRemainingTicks / (1000 * TimeSpan.TicksPerMillisecond), 0) + "s"; 
                double colorMult = (1.0 / TimerColorBaseMult) - ((double)mTimeRemainingTicks / (double)maxTicks);

                if (mTimerOrBonusLabel == null || mTimerOrBonusLabel.Parent != mLayer)
                {
                    mTimerOrBonusLabel = new DrawLabel(text, "Arial", ScoreFontSize * 0.5f, CCLabelFormat.SystemFont);
                    mTimerOrBonusLabel.Color = new CCColor3B(ThemeColors.AdjustColorsForMultipler(colorMult, mTimerOrBonusLabelColorBase));
                    mTimerOrBonusLabel.PositionX = mScoreCircleTop.X;
                    mTimerOrBonusLabel.PositionY = mScoreCircleTop.Y - ScoreFontSize;
                    mTimerOrBonusLabel.HorizontalAlignment = CCTextAlignment.Center;
                    mLayer.AddChild(mTimerOrBonusLabel);
                }
                else if (mTimerOrBonusLabel != null)
                {
                    mTimerOrBonusLabel.Text = text;
                }
            }
        }

        void UpdateGameModeCircle(bool init, bool pulsatMode = false)
        {
            float radius = GameModeCircleRadius;
            double colorAdjust = 1f;

            if (init)
            {
                if (mGameModeCircle != null)
                {
                    mLayer.RemoveChild(mGameModeCircle);
                    mGameModeCircle = null;
                }
                if (mGameModeLabel != null)
                {
                    mLayer.RemoveChild(mGameModeLabel);
                    mGameModeLabel = null;
                }

                mGameModeCircleCenter = new DrawPoint(mLayer.ContentSize.Width * 0.5f, mDrawAreaBounds.MinY);

                mGameModeCircle = new DrawNode();
                mLayer.AddChild(mGameModeCircle);

                DrawColor scoreColor = mThemeColors.TextColor1;
                mGameModeLabel = new DrawLabel("", "Arial", MediumFontSize * 1.15f, CCLabelFormat.SystemFont);
                mGameModeLabel.Color = new CCColor3B(scoreColor);
                mGameModeLabel.PositionX = mGameModeCircleCenter.X;
                mGameModeLabel.PositionY = mGameModeCircleCenter.Y;
                mGameModeLabel.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mGameModeLabel);
            }
            else if (mGameModeCircle == null)
            {
                return;
            }
            else
            {
                mGameModeCircle.Clear();
                mGameModeCircle.Cleanup();
                mGameModeCircle.Visible = false;
            }

            if (pulsatMode)
            {
                colorAdjust *= mPulsate.GetReducedScaleFactor(1 + IncreasedScaleFactorPulsingUI);
            }

            DrawColor color = ThemeColors.AdjustColorsForMultipler(colorAdjust, mThemeColors.AveragedColor13);

            mGameModeCircle.DrawSolidCircle(mGameModeCircleCenter, radius, mThemeColors.BackgroundColor);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius + 1 * mSF, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius - 1 * mSF, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius + 2 * mSF, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius - 2 * mSF, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius + 3 * mSF, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius - 3 * mSF, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius + 4 * mSF, color);
            mGameModeCircle.DrawCircle(mGameModeCircleCenter, radius - 4 * mSF, color);

            mGameModeCircle.Visible = true;

            if (mPlayMode == PlayMode.Standard)
            {
                if (!mLongGameEnabled)
                {
                    mGameModeLabel.Text = "Off";
                }
                else
                {
                    mGameModeLabel.Text = "On";
                }
            }
            if (mPlayMode == PlayMode.Continuity)
            {
                if (!mLongContinuitySeries)
                {
                    mGameModeLabel.Text = "Off";
                }
                else
                {
                    mGameModeLabel.Text = "On";
                }
            }
            else if (mPlayMode == PlayMode.VoidStormV2)
            {
                mGameModeLabel.Text = "+" + (mVoidStormLevelScoreAdjuster * 100);
            }
        }

        void GetBestScores()
        {
            if (mFirstPlay)
            {
                if (mPlayMode == PlayMode.VoidStormV2)
                {
                    mVoidStormBestScore = MainActivity.GetSavedIntWithDefaultValue(0, SharedParams.VoidStormBestScore) / 100.0;
                }
                else if (mPlayMode == PlayMode.Continuity)
                {
                    mContinuitySeriesBestScore = MainActivity.GetSavedIntWithDefaultValue(0, SharedParams.ContinuityBestScore) / 100.0;
                    mContinuityLongSeriesBestScore = MainActivity.GetSavedIntWithDefaultValue(0, SharedParams.ContinuityBestScoreLongSeries) / 100.0;
                }
                else if (mPlayMode == PlayMode.Standard)
                {
                    mBestScore = MainActivity.GetSavedIntWithDefaultValue(0, SharedParams.StandardBestScore) / 100.0;
                    mBestLongGameScore = MainActivity.GetSavedIntWithDefaultValue(0, SharedParams.BestTargetAnglesScore) / 100.0;
                }

                mConseqGamesScoring100Plus = MainActivity.GetSavedIntWithDefaultValue(0, ConseqGames100Plus);
                mBestConseqGamesScoring100Plus = MainActivity.GetSavedIntWithDefaultValue(mConseqGamesScoring100Plus, BestConseqGames100Plus);
                mSumScores100Plus = MainActivity.GetSavedIntWithDefaultValue(0, SumScores100Plus) / 100.0;
            }
        }

        void SetPlayModeLabels()
        {
            string text = "";
            bool headerAtTop = false;
            bool headerAtBottom = false;
            bool headerAtSuperTop = false;
            bool atMiddle = false;
            float headerSizeMult = 1.0f;
            DrawColor headerColor = mThemeColors.Text;

            if (mShowGridSnapMessage && mSnapGridMode)
            {
                atMiddle = true;
                headerSizeMult = 0.967f;

                if (!mNoVirtualization)
                {
                    text += "Normally, the game maps your continuous";
                    text += "\ndrawing to a series of discrete 'moves'.";

                    text += "\n\nWith Snap-to-Grid, you make these";
                    text += "\n'moves' directly, in full chunks.";

                    text += "\n\nThe game is then: \"Minimize directional";
                    text += "\npatterns in your move sequence.\"";
                }
                else
                {
                    text =  "'Pure' mode disables an angle adjustment";
                    text += "\nfeature that makes sharp angles easier.";

                    text += "\n\nThis is a pure way to play, but more";
                    text += "\ncare is needed to mix in sharper angles.";
                }

                mShowGridSnapMessage = false;
            }
            else if (!mReachedTutorialFinishScore && mPlayMode == PlayMode.Standard && mConseqEndRoundEarlyDueToMines >= 2)
            {
                mConseqEndRoundEarlyDueToMines = 0;

                headerAtTop = true;

                text += "\n\nAvoid drawing into boX's.";
                text += "\nTouching the same boX twice ends the game.";
                text += "\n\nInstead, draw into the empty space,";
                text += "\nand minimize repetitive patterns.";
            }
            else if (mPlayMode == PlayMode.Continuity)
            {
                if (mFirstPlay)
                {
                    headerSizeMult = 1f;
                    text += "Play a series of " + mContinuityGamesCount + " seeded games";
                    text += "\n\nThe first is seeded randomly,";
                    text += "\nthe next " + (mContinuityGamesCount - 1) + " seeded by the prior game";
                }

                DrawExtraInfoLabel(
                    info: "Avg: " + Math.Round(mContinuitySeriesScores.Count > 0 ? mContinuitySeriesScores.Average() * 100.0 : 0, 1) + "\nGame: " + (mIsContinuityStarted ? (mContinuitySeriesScores.Count + 1) : 1) + "/" + mContinuityGamesCount + "\n\n",
                    sizeMult1: 1f);
                if (!IsInContinuitySeries())
                {
                    DrawExtraInfoLabel(
                        info2: "Last: " + Math.Round(mContinuityLastSeriesScore * 100.0, 1) + "\nBest: " + Math.Round(mLongContinuitySeries ? mContinuityLongSeriesBestScore : mContinuitySeriesBestScore, 1) + "\n\n",
                        sizeMult2: 1f);
                }
                else if (mContinuitySeriesScores.Count > 0)
                {
                    try
                    {
                        DrawExtraInfoLabel(
                            info2: "Scores: \n" + Math.Round((mContinuitySeriesScores.Count > 0 ? mContinuitySeriesScores[0] : 0) * 100.0, 1) + "  " +
                            Math.Round((mContinuitySeriesScores.Count > 1 ? mContinuitySeriesScores[1] : 0) * 100.0, 1) + "  " +
                            Math.Round((mContinuitySeriesScores.Count > 2 ? mContinuitySeriesScores[2] : 0) * 100.0, 1)
                            + "\n\n",
                            sizeMult2: 1f);
                    } catch { }
                }
            }
            else if (mExperimentMode)
            {
                if (mLastFinalScore == 0)
                {
                    atMiddle = true;

                    text += "Here you play short games, making";
                    text += "\nit easier to see how scoring works.";

                    text += "\n\n\nFirst draw a simple, repeating pattern.";
                    text += "\nThen, gradually introduce complexity.";
                    
                    text += "\n\nObserve how variation effects\nline coloring, brightness, and your score.";

                    text += "\n\n\nSee 'Scoring System' for more info.";
                }
                else
                {
                    headerSizeMult = 0.95f;
                    headerAtBottom = true;
                    text += "Use 'Show Last Draw' for feedback.";
                    text += "\n\nThe graph shows change in curve angle,";
                    text += "\nas both a color gradient, and waveform line.";
                    text += "\nVariation in curve angle is the basis of scoring.";
                }
            }
            else if (mNumberOfFinishedPlaysThisSession == 0)
            {
                if (mPlayMode == PlayMode.VoidStormV2)
                {
                    if (mFirstPlay)
                    {
                        text += "The storm grows continually stronger";
                        text += "\nStay focused on change amidst the chaos";
                    }
                }
                else // Basic game
                {
                    if (mFirstPlay || mBestScore < SharedParams.RankBasic_1Score)
                    {
                        if (mBestScore < SharedParams.RankBasic_1Score)
                        {
                            headerSizeMult = 0.933f;
                            atMiddle = true;

                            text += "\n\n\nThis game is about drawing a continuous curve,";
                            text += "\nwhile doing your best to change how it changes.";

                            text += "\n\nThe obstacle is your own doodling habits:";
                            text += "\nscoring 100 = as good as random scribbling.";

                            text += "\n\n\n\nWe recommend to first vary your scribble";
                            text += "\nin ways intuitive to you, then play the Tutorial.";
                        }
                        else
                        {
                            text += "Draw a continuously changing curve,\nminimizing patterns of change.";
                        }
                    }
                }
            }

            DrawHeaderText(text, headerColor, headerSizeMult, headerAtTop, headerAtBottom, atSuperTop: headerAtSuperTop, atMiddle: atMiddle);
        }

        void DrawHeaderText(string text, DrawColor color, float sizeMult = 1.0f, bool atTop = false, bool atBottom = false, byte alpha = 255, bool atSuperTop = false, bool atMiddle = false)
        {
            mHeaderText = new DrawLabel(text, "Arial", LargeFontSize * sizeMult, CCLabelFormat.SystemFont);
            mHeaderText.Color = new CCColor3B(color.R, color.G, color.B);
            mHeaderText.Opacity = alpha;
            mHeaderText.PositionX = mLayer.ContentSize.Width / 2.0f;
            if (atSuperTop)
            {
                mHeaderText.PositionY = mDrawAreaBounds.MaxY - (LargeFontSize * sizeMult);
            }
            else if (atTop)
            {
                mHeaderText.PositionY = mDrawAreaBounds.MaxY - LargeFontSize * 5 * sizeMult;
            }
            else if (atBottom)
            {
                mHeaderText.PositionY = mDrawAreaBounds.MinY + LargeFontSize * 5 * sizeMult;
            }
            else if (atMiddle)
            {
                mHeaderText.PositionY = mLayer.ContentSize.Height / 2;
            }
            else
            {
                mHeaderText.PositionY = mDrawAreaBounds.MaxY - LargeFontSize * 5 * sizeMult; // For now, same as atTop
            }
            mHeaderText.HorizontalAlignment = CCTextAlignment.Center;
            mLayer.AddChild(mHeaderText);
        }

        void DrawRankLabel(bool pulsate = false)
        {
            if (!mFirstPlay && !mRedrawRankLabel) { return; }

            string rankText = "";
            string bestAndNextText = "";
            float rankFontMult = 1f;

            DrawColor color = mThemeColors.TextColor13;

            if (!pulsate)
            {
                int badgeRank = 0;
                if (mPlayMode == PlayMode.Continuity)
                {
                    badgeRank = 1;
                    if (mContinuityLongSeriesBestScore >= SharedParams.Continuity_4Score)
                    {
                        badgeRank = 5;
                        rankText += SharedParams.Continuity4;
                    }
                    else if (mContinuityLongSeriesBestScore >= SharedParams.Continuity_3Score)
                    {
                        badgeRank = 4;
                        rankText += SharedParams.Continuity3;
                    }
                    else if (mContinuitySeriesBestScore >= SharedParams.Continuity_2Score)
                    {
                        badgeRank = 3;
                        rankText += SharedParams.Continuity2 + ": ";
                        rankText += "Tap to Start";
                    }
                    else if (mContinuitySeriesBestScore >= SharedParams.Continuity_1Score)
                    {
                        badgeRank = 2;
                        rankText += SharedParams.Continuity1 + ": ";
                        rankText += "Tap to Start";
                    }
                }
                else if (mPlayMode == PlayMode.Standard)
                {
                    bestAndNextText = "Best: " + Math.Round(mLongGameEnabled ? mBestLongGameScore : mBestScore, 1);
                    if (mBestLongGameScore >= SharedParams.RankBasic_5Score)
                    {
                        rankText = SharedParams.RankBasic_5;
                        rankFontMult = 1.4f;
                        badgeRank = 5;
                    }
                    else if (mBestLongGameScore >= SharedParams.RankBasic_4Score)
                    {
                        rankText = SharedParams.RankBasic_4;
                        rankFontMult = 1.32f;
                        badgeRank = 4;
                    }
                    else if (mBestScore >= SharedParams.RankBasic_3Score)
                    {
                        rankText = SharedParams.RankBasic_3;
                        rankFontMult = 1.24f;
                        badgeRank = 3;
                    }
                    else if (mBestScore >= SharedParams.RankBasic_2Score)
                    {
                        rankText = SharedParams.RankBasic_2;
                        rankFontMult = 1.16f;
                        badgeRank = 2;
                    }
                    else if (mBestScore >= SharedParams.RankBasic_1Score)
                    {
                        rankText = SharedParams.RankBasic_1;
                        bestAndNextText += "\nGoal: " + 100;
                        rankFontMult = 1.08f;
                        badgeRank = 1;
                    }
                    else
                    {
                        rankText = SharedParams.RankBasic_0;
                        bestAndNextText += "\nGoal: " + 100;
                        rankFontMult = 1.0f;
                    }
                }
                else if (mPlayMode == PlayMode.VoidStormV2)
                {
                    bestAndNextText = "Best: " + Math.Round(mVoidStormBestScore, 1);
                    if (mVoidStormBestScore >= SharedParams.RankVoid_5Score)
                    {
                        rankText = SharedParams.RankVoid_5;
                        rankFontMult = 1.4f;
                        badgeRank = 5;
                    }
                    else if (mVoidStormBestScore >= SharedParams.RankVoid_4Score)
                    {
                        rankText = SharedParams.RankVoid_4;
                        rankFontMult = 1.32f;
                        badgeRank = 4;
                    }
                    else if (mVoidStormBestScore >= SharedParams.RankVoid_3Score)
                    {
                        rankText = SharedParams.RankVoid_3;
                        rankFontMult = 1.24f;
                        badgeRank = 3;
                    }
                    else if (mVoidStormBestScore >= SharedParams.RankVoid_2Score)
                    {
                        rankText = SharedParams.RankVoid_2;
                        rankFontMult = 1.16f;
                        badgeRank = 2;
                    }
                    else
                    {
                        badgeRank = 1;
                        rankText = SharedParams.VoidStorm;
                        rankFontMult = 1.08f;
                    }
                }

                if (mBestAndNextRankLabel != null)
                {
                    mBestAndNextRankLabel.Cleanup();
                    mBestAndNextRankLabel.Dispose();
                    mLayer.RemoveChild(mBestAndNextRankLabel);
                }

                mBestAndNextRankLabel = new DrawLabel(bestAndNextText, "Arial", MediumFontSize * 0.95f, CCLabelFormat.SystemFont);
                mBestAndNextRankLabel.Color = new CCColor3B(mThemeColors.Text);
                mBestAndNextRankLabel.PositionX = mScoreCircleTop.X + (ScoreCircleRadius * 2f);
                mBestAndNextRankLabel.PositionY = mScoreCircleTop.Y - (ScoreCircleRadius * 0.667f);
                mBestAndNextRankLabel.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mBestAndNextRankLabel);

                // When we haven't yet reached 100, don't display a batch or title, so we can show more text on screen.
                if (mPlayMode == PlayMode.Standard && mBestScore < SharedParams.RankBasic_1Score) { return; }

                if (mRankLabel != null)
                {
                    mRankLabel.Cleanup();
                    mRankLabel.Dispose();
                    mLayer.RemoveChild(mRankLabel);
                }

                int badgeRankRadiusLevel = badgeRank;
                if (badgeRankRadiusLevel == 5) { badgeRankRadiusLevel--; }

                mRankBadgeRadius = 1.5f * (float)Math.Sqrt(mDrawAreaBounds.Size.Width * mDrawAreaBounds.Size.Height) / (10 - badgeRankRadiusLevel);
                mRankBadgeCenter = new DrawPoint(mLayer.ContentSize.Width / 2, (mDrawAreaBounds.Size.Height * 0.3f) + mDrawAreaBounds.MinY);
                if (badgeRank == 0)
                {
                    mRankBadgeCenter.Y = (mRankLabel.PositionY - mRankBadgeRadius) - (LargeFontSize * 1.5f);
                }
                if (mLayer.ContentSize.Height / mLayer.ContentSize.Width < SharedParams.SizeAdjustRatioThreshold)
                {
                    float mult = (mLayer.ContentSize.Height / mLayer.ContentSize.Width) / SharedParams.SizeAdjustRatioThreshold;
                    mRankBadgeRadius *= mult;
                }

                float rankTextYPos = mDrawAreaBounds.MidY + (mDrawAreaBounds.Size.Height * 0.1f);
                if (mPlayMode == PlayMode.Continuity && mContinuityLastSeriesScore > 0)
                {
                    // Boost the badge for continuity, and place it in the center
                    mRankBadgeCenter.Y = mDrawAreaBounds.Center.Y;
                    mRankBadgeRadius *= (float)Math.Pow(mContinuityLastSeriesScore, Math.E);
                    rankTextYPos = mDrawAreaBounds.MaxY - (2f * (LargeFontSize * rankFontMult));
                }

                mRankLabel = new DrawLabel(rankText, "Arial", LargeFontSize * rankFontMult, CCLabelFormat.SystemFont);
                mRankLabel.Color = new CCColor3B(color.R, color.G, color.B);
                mRankLabel.PositionX = mLayer.ContentSize.Width / 2.0f;
                mRankLabel.PositionY = rankTextYPos;
                mRankLabel.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mRankLabel);

                mRankBadge = new RankBadge(
                    ref mLayer,
                    mRankBadgeRadius,
                    mRankBadgeCenter,
                    mThemeColors,
                    badgeRank == 0 ? 1 : badgeRank,
                    solidBadge: badgeRank == 0 || mPlayMode == PlayMode.VoidStormV2 || (mPlayMode == PlayMode.Continuity && badgeRank < 4),
                    dotsOnly: badgeRank == 0 || mPlayMode == PlayMode.Continuity,
                    extraLargeDots: badgeRank == 0 || mPlayMode == PlayMode.Continuity);
            }
            else if (mRankLabel != null) // Pulsate
            {
                color = ThemeColors.AdjustColorsForMultipler(mPulsate.GetScaleFactor(), mThemeColors.Color2.R, mThemeColors.Color2.G, mThemeColors.Color2.B);
                mRankLabel.Color = new CCColor3B(color.R, color.G, color.B);

                if (mRankBadge != null)
                {
                    mRankBadge.PulsateBadgeRing(mPulsate.GetScaleFactor(), false);
                }
            }
        }

        void CreateLabels()
        {
            // CLEAN: Make parameters for locations of labels
            SetPlayModeLabels();

            DrawControlLabel();

            UpdateScoreLabel(true);

            if (!mReachedTutorialFinishScore)
            {
                DrawLabel time = new DrawLabel("Draw\nDistance", "Arial", ScoreFontSize * 0.5f, CCLabelFormat.SystemFont);
                time.Color = new CCColor3B(ThemeColors.AdjustColorsForMultipler(0.8, mThemeColors.TextColor2));
                time.PositionX = mScoreCircleTop.X - (ScoreCircleRadius * 1.5f);
                time.PositionY = mScoreCircleTop.Y + (ScoreCircleRadius * 0.933f);
                time.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(time);
            }

            DrawColor darkerAvgColor = ThemeColors.ReduceColorForMultiplierCheap(0.4, mThemeColors.AveragedColor);
            DrawLabel detailsButton = new DrawLabel("?", "Arial", LargeFontSize * 2.2f, CCLabelFormat.SystemFont);
            detailsButton.Color = new CCColor3B(mThemeColors.TextColor1);
            detailsButton.PositionX = mLayer.ContentSize.Width - LargeFontSize * 1.333f;
            detailsButton.PositionY = mLayer.ContentSize.Height - LargeFontSize * 1.333f;
            detailsButton.HorizontalAlignment = CCTextAlignment.Center;

            DrawColor underlayOutlineColor = mThemeColors.Color1;
            DrawNode detailsButtonUnderlay = new DrawNode();
            detailsButtonUnderlay.DrawSolidCircle(detailsButton.Position, LargeFontSize, darkerAvgColor);
            detailsButtonUnderlay.DrawCircle(detailsButton.Position, LargeFontSize, underlayOutlineColor);
            mLayer.AddChild(detailsButtonUnderlay);
            mLayer.AddChild(detailsButton);

            DrawLabel back = new DrawLabel(Convert.ToChar(8592).ToString(), "Arial", LargeFontSize * 2.8f, CCLabelFormat.SystemFont);
            back.Color = new CCColor3B(mThemeColors.TextColor1);
            back.PositionX = LargeFontSize * 1.333f;
            back.PositionY = mLayer.ContentSize.Height - LargeFontSize * 0.8333f;
            back.HorizontalAlignment = CCTextAlignment.Center;

            DrawNode backUnderlay = new DrawNode();
            backUnderlay.DrawSolidCircle(new DrawPoint(back.Position.X, back.PositionY - (LargeFontSize / 2)), LargeFontSize, darkerAvgColor);
            backUnderlay.DrawCircle(new DrawPoint(back.Position.X, back.PositionY - (LargeFontSize / 2)), LargeFontSize, mThemeColors.Color1);
            mLayer.AddChild(backUnderlay);
            mLayer.AddChild(back);

            {
                double colorMult = !mSnapGridMode ? 1.0 : 1.2;
                mSnapGridButton = new DrawLabel("", "Arial", MediumFontSize, CCLabelFormat.SystemFont);
                mSnapGridButton.Color = new CCColor3B(ThemeColors.AdjustColorsForMultipler(colorMult, mThemeColors.TextColor1));
                mSnapGridButton.PositionX = mLayer.ContentSize.Width * 0.15f;
                mSnapGridButton.PositionY = mDrawAreaBounds.MinY * 0.5f;
                mSnapGridButton.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mSnapGridButton);

                string text = "Snap-to-Grid\n";
                if (mSnapGridMode) { text += "On"; }
                else { text += "Off"; }
                mSnapGridButton.Text = text;
            }
            if (mSnapGridMode)
            {
                double colorMult = !mNoVirtualization ? 0.8 : 1.2;
                mPureModeButton = new DrawLabel("Pure", "Arial", MediumFontSize*0.9f, CCLabelFormat.SystemFont);
                mPureModeButton.Color = new CCColor3B(ThemeColors.AdjustColorsForMultipler(colorMult, mThemeColors.TextColor1));
                mPureModeButton.PositionX = mLayer.ContentSize.Width * 0.0833f;
                mPureModeButton.PositionY = mDrawAreaBounds.MinY * 0.1f;
                mPureModeButton.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mPureModeButton);

                string text = "Pure:";
                if (mNoVirtualization) { text += "On"; }
                else { text += "Off"; }
                mPureModeButton.Text = text;
            }

            if (mPlayMode == PlayMode.Standard)
            {
                // Tutorial related
                float fontSize = MediumFontSize;
                double colorMult = 1.1;

                if (mBestScore < 100)
                {
                    fontSize *= 1.2f;
                }

                mExperimentModeToggle = new DrawLabel("", "Arial", fontSize, CCLabelFormat.SystemFont);
                mExperimentModeToggle.Color = new CCColor3B(ThemeColors.AdjustColorsForMultipler(colorMult, mThemeColors.TextColor1));
                mExperimentModeToggle.PositionX = mLayer.ContentSize.Width * 0.75f;
                mExperimentModeToggle.PositionY = mLayer.ContentSize.Height * 0.96f;
                mExperimentModeToggle.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mExperimentModeToggle);

                if (mExperimentMode)
                {
                    mExperimentModeToggle.Text = "Exit\nTutorial";

                    DrawLabel scoreSys = new DrawLabel("Scoring\nSystem", "Arial", MediumFontSize*1.1f, CCLabelFormat.SystemFont);
                    scoreSys.Color = new CCColor3B(ThemeColors.AdjustColorsForMultipler(1.2, mThemeColors.TextColor1));
                    scoreSys.PositionX = mLayer.ContentSize.Width * 0.25f;
                    scoreSys.PositionY = mLayer.ContentSize.Height * 0.9f;
                    scoreSys.HorizontalAlignment = CCTextAlignment.Center;
                    mLayer.AddChild(scoreSys);
                }
                else
                {
                    mExperimentModeToggle.Text = "Play\nTutorial";
                }
            }
        }

        void UpdateScoreLabel(bool init = false)
        {
            if (init)
            {
                if (mScoreLabel != null)
                {
                    mLayer.RemoveChild(mScoreLabel);
                    mScoreLabel = null;
                }

                string score = "0.0";
                if (mLastFinalScore != 0)
                {
                    score = (100 * Math.Round(mLastFinalScore, 3)).ToString("0.0");
                }
                if (mLastFinalScoreBonus != 0 && mTimerOrBonusLabel != null)
                {
                    if (mLastFinalScoreBonus < 0)
                    {
                        mTimerOrBonusLabel.Text = "-" + Math.Round(100 * -mLastFinalScoreBonus, 0);
                    }
                    else
                    {
                        mTimerOrBonusLabel.Text = "+" + Math.Round(100 * mLastFinalScoreBonus, 0);
                    }
                }

                mScoreLabel = new DrawLabel(score, "Arial", ScoreFontSize, CCLabelFormat.SystemFont);
                mScoreLabel.Color = new CCColor3B(mThemeColors.TextColor13);
                mScoreLabel.PositionX = mScoreCircleTop.X;
                mScoreLabel.PositionY = mScoreCircleTop.Y;
                mScoreLabel.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mScoreLabel);
            }
            else if (mScoreFeedbackHidden)
            {
                mScoreLabel.Text = "Hidden";
            }
            else
            {
                mScoreLabel.Text = "" + (100 * Math.Round(mFinalScore, 3)).ToString("0.0");
            }
        }

        const string ShowLastDraw = "Show Last Draw";
        void DrawControlLabel()
        {
            if (mControlLabel != null)
            {
                mLayer.RemoveChild(mControlLabel);
                mControlLabel = null;
            }

            const float lableSizeAdjuster = 1.333f;
            DrawColor brightColor1 = ThemeColors.AdjustColorsForMultipler(1.2, mThemeColors.TextColor1);
            mControlLabel = new DrawLabel("", "Arial", MediumFontSize*lableSizeAdjuster, CCLabelFormat.SystemFont);
            mControlLabel.Color = new CCColor3B(brightColor1);
            mControlLabel.PositionX = mLayer.ContentSize.Width * 0.5f;
            mControlLabel.PositionY = mDrawAreaBounds.MinY / 5f;
            mControlLabel.HorizontalAlignment = CCTextAlignment.Center;
            mLayer.AddChild(mControlLabel);

            if ((mPlayMode != PlayMode.Continuity || mContinuityScoredDrawStartIndex > 0 || mContinuityScoredDrawEndIndex > 0 || !IsInContinuitySeries()) &&
                mScoredDraws.Count > MinMovesToTriggerEnd)
            {
                mControlLabel.Text = ShowLastDraw; 
            }
            else if (mPlayMode == PlayMode.VoidStormV2 && !mReadyForDrawing)
            {
                mControlLabel.Color = new CCColor3B(mThemeColors.Text);
                if (mVoidStormLevel == 1)
                {
                    mControlLabel.Text = SharedParams.StormLevel1;
                }
                else if (mVoidStormLevel == 2)
                {
                    mControlLabel.Text = SharedParams.StormLevel2;
                }
                else if (mVoidStormLevel == 3)
                {
                    mControlLabel.Text = SharedParams.StormLevel3;
                }
                else if (mVoidStormLevel >= 4)
                {
                    mControlLabel.Text = SharedParams.StormLevel4;
                }
                else
                {
                    mControlLabel.Text = SharedParams.StormLevel0;
                }
                mControlLabel.Text += " (" + mVoidStormLevel + ")";
            }
            else if ((mLongGameEnabled || mLongContinuitySeries) && !mReadyForDrawing)
            {
                mControlLabel.Color = new CCColor3B(mThemeColors.Text);
                mControlLabel.Text = mPlayMode == PlayMode.Standard ? SharedParams.TargetAnglesRing : SharedParams.LongContinuitySeries;
            }
        }

        void DrawExtraInfoLabel(string info = null, string info2 = null, float sizeMult1 = 0.9f, float sizeMult2 = 0.9f, bool clickable = false, DrawColor? textColor = null)
        {
            DrawColor color;
            if (textColor != null)
            {
                color = textColor.Value;
            }
            else
            {
                color = mThemeColors.TextColor1;
            }

            if (info != null)
            {
                if (mExtraInfoLabel != null)
                {
                    mLayer.RemoveChild(mExtraInfoLabel);
                    mExtraInfoLabel = null;
                }

                mExtraInfoLabel = new DrawLabel(info, "Arial", MediumFontSize * sizeMult1, CCLabelFormat.SystemFont);
                mExtraInfoLabel.Color = clickable ? new CCColor3B(color) : new CCColor3B(ThemeColors.AdjustColorsForMultipler(1.1, color));
                mExtraInfoLabel.PositionX = 0.15f * mDrawAreaBounds.MaxX;
                mExtraInfoLabel.PositionY = mDrawAreaBounds.MaxY + (25 * mSF);
                mExtraInfoLabel.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mExtraInfoLabel);
            }

            if (info2 != null)
            {
                if (mExtraInfoLabel2 != null)
                {
                    mLayer.RemoveChild(mExtraInfoLabel2);
                    mExtraInfoLabel2 = null;
                }

                mExtraInfoLabel2 = new DrawLabel(info2, "Arial", MediumFontSize * sizeMult2, CCLabelFormat.SystemFont);
                mExtraInfoLabel2.Color = clickable ? new CCColor3B(color) : new CCColor3B(ThemeColors.AdjustColorsForMultipler(1.1, color));
                mExtraInfoLabel2.PositionX = mDrawAreaBounds.MaxX * 0.825f;
                mExtraInfoLabel2.PositionY = mDrawAreaBounds.MaxY + (25 * mSF);
                mExtraInfoLabel2.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(mExtraInfoLabel2);
            }
        }

        void ShowDetailsOverlay()
        {
            DrawPoint start;
            DrawPoint end;
            start = new DrawPoint(0, 0);
            end = new DrawPoint(mLayer.ContentSize.Width, 0);

            mDetailsOverlay = new DrawNode();
            mLayer.AddChild(mDetailsOverlay);
            mDetailsOverlay.DrawLine(
                from: start,
                to: end,
                color: mThemeColors.OverlayDark,
                lineWidth: mLayer.ContentSize.Height);

            string detailsText = "";

            string rank1Text = "";
            string rank1TextCriteria = "";
            string rank2Text = "";
            string rank2TextCriteria = "";
            string rank3Text = "";
            string rank3TextCriteria = "";
            string rank4Text = "";
            string rank4TextCriteria = "";
            string rank5Text = "";
            string rank5TextCriteria = "";

            float yOffset = 0;
            float yOffsetDetails = 0;
            float badgeSizeMult = 1;
            float textSizeMult = 1;

            int ranksCount = 0;

            if (mPlayMode == PlayMode.Continuity)
            {
                detailsText += "Your score is averaged over a " + mContinuityGamesCount + " game series.";

                detailsText += "\n\nThe initial game is seeded randomly.";
                detailsText += "\n\nThe next games retain the highest, or lowest,";
                detailsText += "\nscoring " + mSeededMovesContinuity + " moves from the prior game -";
                detailsText += "\nwhichever was more prominent in absolute value.";

                rank1Text = SharedParams.Continuity1;
                rank1TextCriteria = "Average " + SharedParams.Continuity_1Score;

                rank2Text = SharedParams.Continuity2;
                rank2TextCriteria = "\nAverage " + SharedParams.Continuity_2Score;
                rank2TextCriteria += "\n◉  Unlock:  " + SharedParams.LongContinuitySeries + "  ◉";

                rank3Text = SharedParams.Continuity3;
                rank3TextCriteria = "Average " + SharedParams.Continuity_3Score + " in a Long Series";

                rank4Text = SharedParams.Continuity4;
                rank4TextCriteria = "Average " + SharedParams.Continuity_4Score + " in a Long Series";

                textSizeMult = 0.95f;
                yOffsetDetails = 0.02f;
                yOffset = MediumFontSize * 6f;
                badgeSizeMult = 1.15f;
                ranksCount = 4;
            }
            if (mPlayMode == PlayMode.Standard)
            {
                detailsText = "Vary your free-form drawing in ways intuitive to you.";

                detailsText += "\n\nDraw distance is " + mMovesPerRound + " grid units.";
                if (!mExperimentMode)
                {
                    detailsText += "\nTime limit is " + Math.Round((mMovesPerRound * (mLongGameEnabled ? MSPerMoveTimeLimitLongGame : MSPerMoveTimeLimit)) / 1000, 0) + " seconds.";
                }

                detailsText += "\n\nBoX's alter the draw space, changing each game.";
                detailsText += "\nTouching the same boX twice ends the round.";

                yOffset = MediumFontSize * 5f;
                yOffsetDetails = -0.025f;
                badgeSizeMult = 0.95f;
                textSizeMult = 0.95f;

                rank1Text = SharedParams.RankBasic_1;
                rank1TextCriteria = "Score " + SharedParams.RankBasic_1Score;

                rank2Text = SharedParams.RankBasic_2;
                rank2TextCriteria = "Score " + SharedParams.RankBasic_2Score;

                rank3Text = SharedParams.RankBasic_3;
                rank3TextCriteria = "\nScore " + SharedParams.RankBasic_3Score;
                rank3TextCriteria += "\n◉  Unlock:  " + SharedParams.TargetAnglesRing + "  ◉";

                rank4Text = SharedParams.RankBasic_4;
                rank4TextCriteria = "Score " + SharedParams.RankBasic_4Score + " in a Long Game";

                rank5Text = SharedParams.RankBasic_5;
                rank5TextCriteria = "Score " + SharedParams.RankBasic_5Score + " in a Long Game";

                ranksCount = 5;
            }
            else if (mPlayMode == PlayMode.VoidStormV2)
            {
                detailsText += "Scoring works same as standard Mind Stream.";
                detailsText += "\nBoX's spawn and decay at a constant rate, and";
                detailsText += "\nspawn larger (up to 2x) as the game goes on.";

                detailsText += "\n\nTouching an active boX gives -5 points.";

                detailsText += "\n\nStorm levels give +10 bonus points per level.";
                detailsText += "\nSpawn rate gets +30%\\lvl; game length +30\\lvl.";

                yOffset = MediumFontSize * 6f;
                yOffsetDetails = 0f;
                badgeSizeMult = 1f;
                textSizeMult = 0.95f;

                rank1Text = SharedParams.RankVoid_2;
                rank1TextCriteria = "\nScore " + SharedParams.RankVoid_2Score;
                rank1TextCriteria += "\nUnlock Storm Level 1:  " + SharedParams.StormLevel1;

                rank2Text = SharedParams.RankVoid_3;
                rank2TextCriteria = "\nScore " + SharedParams.RankVoid_3Score;
                rank2TextCriteria += "\nUnlock Storm Level 2:  " + SharedParams.StormLevel2;

                rank3Text = SharedParams.RankVoid_4;
                rank3TextCriteria = "\nScore " + SharedParams.RankVoid_4Score;
                rank3TextCriteria += "\nUnlock Storm Level 3:  " + SharedParams.StormLevel3;

                rank4Text = SharedParams.RankVoid_5;
                rank4TextCriteria = "\nScore " + SharedParams.RankVoid_5Score;
                rank4TextCriteria += "\nUnlock Storm Level 4+:  " + SharedParams.StormLevel4;

                ranksCount = 4;
            }

            float rankTitleSize = MediumFontSize * 1.333f * badgeSizeMult;
            float criteriaFontSize = MediumFontSize * 1.111f * badgeSizeMult;

            float centerX = mLayer.ContentSize.Width / 2;

            DrawLabel labelInstr = new DrawLabel(detailsText, "Arial", MediumFontSize * textSizeMult, CCLabelFormat.SystemFont);
            labelInstr.Color = new CCColor3B(mThemeColors.Text.R, mThemeColors.Text.G, mThemeColors.Text.B);
            labelInstr.PositionX = centerX;
            labelInstr.PositionY = mLayer.ContentSize.Height * (0.87f - yOffsetDetails);
            labelInstr.HorizontalAlignment = CCTextAlignment.Center;
            mLayer.AddChild(labelInstr);
            mDetailsTexts.Add(labelInstr);

            // Draw faded versions of the title badges in the background, so they look like they are in the background
            float radiusBase = MediumFontSize * 2.8f * badgeSizeMult;
            int badgeLevelAdd = 5 - ranksCount;
            mDetailsBadges.Add(new RankBadge(
                ref mLayer,
                radiusBase * 0.7f,
                new DrawPoint(centerX, labelInstr.PositionY - (yOffset + MediumFontSize * 4f)),
                mThemeColors,
                1 + badgeLevelAdd,
                mPlayMode == PlayMode.VoidStormV2, true, true,
                mPlayMode == PlayMode.Continuity));
            mDetailsBadges.Add(new RankBadge(
                ref mLayer,
                radiusBase * 0.8f,
                new DrawPoint(centerX, labelInstr.PositionY - (yOffset + MediumFontSize * 11f)),
                mThemeColors,
                2 + badgeLevelAdd,
                mPlayMode == PlayMode.VoidStormV2, true, true,
                mPlayMode == PlayMode.Continuity));
            mDetailsBadges.Add(new RankBadge(
                ref mLayer,
                radiusBase * 0.9f,
                new DrawPoint(centerX, labelInstr.PositionY - (yOffset + MediumFontSize * 18f)),
                mThemeColors,
                3 + badgeLevelAdd,
                mPlayMode == PlayMode.VoidStormV2, true, true,
                mPlayMode == PlayMode.Continuity));
            if (4 + badgeLevelAdd <= 5)
            {
                mDetailsBadges.Add(new RankBadge(
                    ref mLayer,
                    radiusBase,
                    new DrawPoint(centerX, labelInstr.PositionY - (yOffset + MediumFontSize * 25f)),
                    mThemeColors,
                    4 + badgeLevelAdd,
                    mPlayMode == PlayMode.VoidStormV2, true, true,
                    mPlayMode == PlayMode.Continuity));
            }
            if (5 + badgeLevelAdd <= 5)
            {
                mDetailsBadges.Add(new RankBadge(
                    ref mLayer,
                    radiusBase,
                    new DrawPoint(centerX, labelInstr.PositionY - (yOffset + MediumFontSize * 32f)),
                    mThemeColors,
                    5 + badgeLevelAdd,
                    mPlayMode == PlayMode.VoidStormV2, true, true,
                    mPlayMode == PlayMode.Continuity));
            }

            // List the title labels on top of the badges
            if (ranksCount >= 1)
            {
                DrawLabel labelRank1 = new DrawLabel(rank1Text, "Arial", rankTitleSize, CCLabelFormat.SystemFont);
                labelRank1.Color = new CCColor3B(mThemeColors.TextColor2.R, mThemeColors.TextColor2.G, mThemeColors.TextColor2.B);
                labelRank1.PositionX = centerX;
                labelRank1.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 3.5f);
                labelRank1.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelRank1);
                mDetailsTexts.Add(labelRank1);
                DrawLabel labelCrit1 = new DrawLabel(rank1TextCriteria, "Arial", criteriaFontSize, CCLabelFormat.SystemFont);
                labelCrit1.Color = new CCColor3B(mThemeColors.Text);
                labelCrit1.PositionX = centerX;
                labelCrit1.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 5f);
                labelCrit1.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelCrit1);
                mDetailsTexts.Add(labelCrit1);
            }
            if (ranksCount >= 2)
            {
                DrawLabel labelRank2 = new DrawLabel(rank2Text, "Arial", rankTitleSize, CCLabelFormat.SystemFont);
                labelRank2.Color = new CCColor3B(mThemeColors.TextColor2.R, mThemeColors.TextColor2.G, mThemeColors.TextColor2.B);
                labelRank2.PositionX = centerX;
                labelRank2.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 10.5f);
                labelRank2.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelRank2);
                mDetailsTexts.Add(labelRank2);
                DrawLabel labelCrit2 = new DrawLabel(rank2TextCriteria, "Arial", criteriaFontSize, CCLabelFormat.SystemFont);
                labelCrit2.Color = new CCColor3B(mThemeColors.Text);
                labelCrit2.PositionX = centerX;
                labelCrit2.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 12f);
                labelCrit2.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelCrit2);
                mDetailsTexts.Add(labelCrit2);
            }
            if (ranksCount >= 3)
            {
                DrawLabel labelRank3 = new DrawLabel(rank3Text, "Arial", rankTitleSize, CCLabelFormat.SystemFont);
                labelRank3.Color = new CCColor3B(mThemeColors.TextColor2.R, mThemeColors.TextColor2.G, mThemeColors.TextColor2.B);
                labelRank3.PositionX = centerX;
                labelRank3.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 17.5f);
                labelRank3.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelRank3);
                mDetailsTexts.Add(labelRank3);
                DrawLabel labelCrit3 = new DrawLabel(rank3TextCriteria, "Arial", criteriaFontSize, CCLabelFormat.SystemFont);
                labelCrit3.Color = new CCColor3B(mThemeColors.Text);
                labelCrit3.PositionX = centerX;
                labelCrit3.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 19f);
                labelCrit3.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelCrit3);
                mDetailsTexts.Add(labelCrit3);
            }
            if (ranksCount >= 4)
            {
                DrawLabel labelRank4 = new DrawLabel(rank4Text, "Arial", rankTitleSize, CCLabelFormat.SystemFont);
                labelRank4.Color = new CCColor3B(mThemeColors.TextColor2.R, mThemeColors.TextColor2.G, mThemeColors.TextColor2.B);
                labelRank4.PositionX = centerX;
                labelRank4.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 24.5f);
                labelRank4.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelRank4);
                mDetailsTexts.Add(labelRank4);
                DrawLabel labelCrit4 = new DrawLabel(rank4TextCriteria, "Arial", criteriaFontSize, CCLabelFormat.SystemFont);
                labelCrit4.Color = new CCColor3B(mThemeColors.Text);
                labelCrit4.PositionX = centerX;
                labelCrit4.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 26f);
                labelCrit4.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelCrit4);
                mDetailsTexts.Add(labelCrit4);
            }
            if (ranksCount >= 5)
            { 
                DrawLabel labelRank5 = new DrawLabel(rank5Text, "Arial", rankTitleSize, CCLabelFormat.SystemFont);
                labelRank5.Color = new CCColor3B(mThemeColors.TextColor2.R, mThemeColors.TextColor2.G, mThemeColors.TextColor2.B);
                labelRank5.PositionX = centerX;
                labelRank5.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 31.5f);
                labelRank5.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelRank5);
                mDetailsTexts.Add(labelRank5);
                DrawLabel labelCrit5 = new DrawLabel(rank5TextCriteria, "Arial", criteriaFontSize, CCLabelFormat.SystemFont);
                labelCrit5.Color = new CCColor3B(mThemeColors.Text);
                labelCrit5.PositionX = centerX;
                labelCrit5.PositionY = labelInstr.PositionY - (yOffset + MediumFontSize * 33f);
                labelCrit5.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(labelCrit5);
                mDetailsTexts.Add(labelCrit5);
            }

            if (mPlayMode != PlayMode.VoidStormV2)
            {
                DrawLabel conseq100PlusAvg = new DrawLabel("", "Arial", MediumFontSize, CCLabelFormat.SystemFont);
                conseq100PlusAvg.Color = new CCColor3B(mThemeColors.TextColor2);
                conseq100PlusAvg.PositionX = mLayer.ContentSize.Width * 0.5f;
                conseq100PlusAvg.PositionY = mLayer.ContentSize.Height * 0.0333f;
                conseq100PlusAvg.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(conseq100PlusAvg);
                mDetailsTexts.Add(conseq100PlusAvg);

                if (mConseqGamesScoring100Plus > 0)
                {
                    conseq100PlusAvg.Text = "Avg score over last " + mConseqGamesScoring100Plus + " games: " + Math.Round((mSumScores100Plus / mConseqGamesScoring100Plus), 1);
                }
            }
        }

        void RemoveTextUnderlay()
        {
            if (mTextUnderlay != null)
            {
                mTextUnderlay.Visible = false;
                mLayer.RemoveChild(mTextUnderlay);
                mTextUnderlay = null;
            }
        }

        void DrawTextUnderlay(float y1, float y2)
        {
                mTextUnderlay = new DrawNode();
                mTextUnderlay.DrawLine(
                    from: new DrawPoint(mLayer.ContentSize.Width / 2, y1),
                    to: new DrawPoint(mLayer.ContentSize.Width / 2, y2),
                    color: new DrawColor(0, 0, 0, 128),
                    lineWidth: mLayer.ContentSize.Width / 2);
                mLayer.AddChild(mTextUnderlay);
            }

        void RemoveDetailsOverlay()
        {
            if (mDetailsOverlay != null)
            {
                mLayer.RemoveChild(mDetailsOverlay);
                mDetailsOverlay = null;
            }
            if (mDetailsTexts != null)
            {
                for (int j = 0; j < mDetailsTexts.Count; j++)
                {
                    mLayer.RemoveChild(mDetailsTexts[j]);
                    mDetailsTexts[j] = null;
                }

                mDetailsTexts.Clear();
            }
            if (mDetailsBadges != null)
            {
                for (int j = 0; j < mDetailsBadges.Count; j++)
                {
                    mDetailsBadges[j].Deactivate();
                    mDetailsBadges[j] = null;
                }
                mDetailsBadges.Clear();
            }
        }

        int MinesToSpawn()
        {
            long now = DateTime.UtcNow.Ticks;
            long elapsed = now - mLastMineSpawnTicks;
            int count = 0;

            while (elapsed >= mVoidSpawnRateTicks)
            {
                elapsed -= mVoidSpawnRateTicks;
                count++;
            }
            mLastMineSpawnTicks = now - elapsed;

            return count;
        }

        int MinesToRemove()
        {
            long now = DateTime.UtcNow.Ticks;
            long elapsed = now - mLastMineRemoveTicks;
            int count = 0;

            while (elapsed >= mVoidRemoveRateTicks)
            {
                elapsed -= mVoidRemoveRateTicks;
                count++;
            }
            mLastMineRemoveTicks = now - elapsed;

            return count;
        }

        int GetMaxVoidStormLevel()
        {
            if (mVoidStormBestScore >= SharedParams.StormLevel4Score)
            {
                return 1 + (int)((mVoidStormBestScore - 100) / 10.0);
            }
            else if (mVoidStormBestScore >= SharedParams.StormLevel3Score)
            {
                return 3;
            }
            else if (mVoidStormBestScore >= SharedParams.StormLevel2Score)
            {
                return 2;
            }
            else if (mVoidStormBestScore >= SharedParams.StormLevel1Score)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        // Sets the maximum sub-mode state. Used to auto-enable sub game modes after new achievements. 
        void SetMaxSubModeState()
        {
            if (mPlayMode == PlayMode.VoidStormV2)
            {
                mVoidStormLevel = GetMaxVoidStormLevel();
                MainActivity.SaveIntValue(mVoidStormLevel, VoidStormOrbLevel);
            }
        }

        void ToggleExperimentMode()
        {
            mFirstPlay = false;
            MainActivity.SoundEffects.MenuItem();
            mExperimentMode = !mExperimentMode;
            mTutorialMovesPerRoundIndex = 0;
            mMineManager.EnableDisableMines(mExperimentMode);
        }

        void UpdateVoidStormTimerOrBonusLabel()
        {
            if (mTimerOrBonusLabel != null && mLastFinalScoreBonus == 0)
            { 
                if (mVoidStormLevelScoreAdjuster < 0)
                {
                    mTimerOrBonusLabel.Text = "-" + Math.Round(100 * -mVoidStormLevelScoreAdjuster, 0);
                }
                else
                {
                    mTimerOrBonusLabel.Text = "+" + Math.Round(100 * mVoidStormLevelScoreAdjuster, 0);
                }
            }
        }

        void DrawShowGridMovesTextandOverlay()
        {
            if (mTextUnderlay == null)
            {
                DrawTextUnderlay(mDrawAreaBounds.MinY - LargeFontSize * 4.0f, mDrawAreaBounds.MinY + LargeFontSize * 4.0f);
            }

            DrawHeaderText(
                "\n\n\n\n\n\n\n\nAngles shown are what the scoring AI processes." +
                "\nEach angle is averaged over 1 grid unit of drawing." +

                "\n\n\n\nThe grid optimizes each game; fix it to the Max below.",
                mThemeColors.Text,
                sizeMult: 0.8f,
                atBottom: true);
        }

        DrawPoint mLastTouchUpPausePoint;
        private void HandleTouchesBegan(List<CCTouch> touches, CCEvent touchEvent)
        {        
            if (mReadyForDrawing && touches[0].Id != 0 && !mIsFingerPausedOrLifted) { return; }

            DrawPoint location = touches[0].Location;

            if (mMovesPerRound != mMovesRemaining)
            {
                // We are currently drawing
                if (!mDrawAreaBounds.ContainsPoint(location))
                {
                    MainActivity.SoundEffects.RoundEndEarly();
                    mEndRoundEarly = true;
                }
                else
                {
                    DrawTraceLine(mLastTouchUpPausePoint.X, mLastTouchUpPausePoint.Y, location.X, location.Y);

                    bool hit = mMineManager.HitTestMinesBetweenPoints(location, mLastTouchUpPausePoint);
                    // Hit test mines in between
                    if (mPlayMode == PlayMode.VoidStormV2)
                    {
                        if (hit) { mVoidStormLevelScoreAdjuster -= VoidStormScoreBonusDecrementPerHit; }
                    }
                    else
                    {
                        mEndRoundEarly = hit;
                        if (mEndRoundEarly)
                        {
                            mConseqEndRoundEarlyDueToMines++;
                        }
                    }
                }

                return;
            }

            // CLEAN: Use parameters for all UI locations
            if (mDetailsOverlay != null)
            {
                RemoveDetailsOverlay();
                mShowingSpecialDetails = false;
                return;
            }
            if (location.X >= mLayer.ContentSize.Width * 0.88f && location.Y >= mLayer.ContentSize.Height * 0.93f)
            {
                ShowDetailsOverlay();
                mShowingSpecialDetails = true;
                return;
            }
            if (location.X <= mLayer.ContentSize.Width * 0.14f && location.Y >= mLayer.ContentSize.Height * 0.93f)
            {
                GameController.PopScene();
                return;
            }
            if (mPlayMode == PlayMode.Standard &&
                location.X >= mLayer.ContentSize.Width * 0.68 && location.X <= mLayer.ContentSize.Width * 0.82 &&
                location.Y >= mLayer.ContentSize.Height * 0.92f)
            {
                ToggleExperimentMode();
                StartNewRound();
                return;
            }
            if (mExperimentMode &&
                location.X >= mLayer.ContentSize.Width * 0.18 && location.X <= mLayer.ContentSize.Width * 0.32 &&
                location.Y >= mLayer.ContentSize.Height * 0.86f && location.Y <= mLayer.ContentSize.Height * 0.94f)
            {
                MainActivity.Current.ShowHelp();
                return;
            }
            if (Math.Pow(location.X - mScoreCircleTop.X, 2) + Math.Pow(location.Y - mScoreCircleTop.Y, 2) < Math.Pow(ScoreCircleRadius, 2))
            {
                MainActivity.SoundEffects.MenuItem();
                if (mExperimentMode)
                {
                    mTutorialMovesPerRoundIndex++;
                    if (mTutorialMovesPerRoundIndex > TutorialMovesPerRoundCycle.Length)
                    {
                        mTutorialMovesPerRoundIndex = 0;
                    }
                    StartNewRound();
                }
                else
                {
                    mScoreFeedbackHidden = !mScoreFeedbackHidden;
                    if (mScoreFeedbackHidden)
                    {
                        ScoreFontSize = LargeFontSize;
                    }
                    else
                    {
                        ScoreFontSize = ScoreFontSizeBase * ScoreCircleRadius;
                    }
                    UpdateScoreLabel(true);
                    UpdateScoreLabel();
                }
                return;
            }
            if (Math.Pow(location.X - mGameModeCircleCenter.X, 2) + Math.Pow(location.Y - mGameModeCircleCenter.Y, 2) < Math.Pow(GameModeCircleRadius, 2))
            {
                if ((mPlayMode == PlayMode.Standard && mBestScore >= SharedParams.RoBMUnlockBaseScore)
                    || (mPlayMode == PlayMode.Continuity && !IsInContinuitySeries() && mContinuitySeriesBestScore >= SharedParams.ContinuityLongSeriesUnlock))
                {
                    MainActivity.SoundEffects.MenuItem();

                    if (mPlayMode == PlayMode.Standard) 
                    { 
                        mLongGameEnabled = !mLongGameEnabled;
                        MainActivity.SaveIntValue(mLongGameEnabled ? 1 : 0, TargetAnglesEnabled);
                    }
                    else if (mPlayMode == PlayMode.Continuity) 
                    { 
                        mLongContinuitySeries = !mLongContinuitySeries;
                        MainActivity.SaveIntValue(mLongContinuitySeries ? 1 : 0, LongSeriesEnabled);
                        mInitializedContinuityScoredDrawStartIndex = false;
                    }

                    ClearAllLines(true);
                    mScoredDraws.Clear(); // We do this so that the Ring state from the last game in indicated in Show Last Draw

                    mResetMineCenters = true;

                    StartNewRound();
                    return;
                }
            }
            if (mPlayMode == PlayMode.VoidStormV2 && mVoidStormBestScore >= SharedParams.StormLevel1Score &&
                Math.Pow(location.X - mDrawAreaBounds.MidX, 2) + Math.Pow(location.Y - mDrawAreaBounds.MinY, 2) < Math.Pow(GameModeCircleRadius, 2))
            {
                MainActivity.SoundEffects.MenuItem();
                int maxOrbLevel = GetMaxVoidStormLevel();
                if (maxOrbLevel > 0)
                {
                    mVoidStormLevel++;
                    if (mVoidStormLevel > maxOrbLevel)
                    {
                        mVoidStormLevel = 0;
                    }
                    MainActivity.SaveIntValue(mVoidStormLevel, VoidStormOrbLevel);
                }

                ClearAllLines(true);
                mScoredDraws.Clear();

                mResetMineCenters = false;
                StartNewRound();
                return;
            }
            if (location.X < mLayer.ContentSize.Width * 0.222f && location.X > 0 &&
                location.Y < mDrawAreaBounds.MinY * 0.75 && location.Y > mDrawAreaBounds.MinY * 0.25)
            {
                if (!IsInContinuitySeries())
                {
                    MainActivity.SoundEffects.MenuItem();

                    mSnapGridMode = !mSnapGridMode;

                    MainActivity.SaveIntValue(mSnapGridMode ? 1 : 0, SnapToGridMode);

                    if (mSnapGridMode) { mShowGridSnapMessage = true; }
                    else { mNoVirtualization = false; }

                    mFirstPlay = false;
                    StartNewRound();
                    return;
                }
            }
            if (mSnapGridMode &&
                location.X < mLayer.ContentSize.Width * 0.11 && location.X > mLayer.ContentSize.Width * 0.01 &&
                location.Y < mLayer.ContentSize.Height * 0.11 && location.Y > mLayer.ContentSize.Height * 0.01)
            {
                if (!IsInContinuitySeries())
                {
                    MainActivity.SoundEffects.MenuItem();

                    mNoVirtualization = !mNoVirtualization;

                    if (mNoVirtualization) { mShowGridSnapMessage = true; }

                    mFirstPlay = false;
                    StartNewRound();
                    return;
                }
            }
            if ((location.X > mLayer.ContentSize.Width * 0.825 && location.X < mLayer.ContentSize.Width &&
                location.Y < mDrawAreaBounds.MinY * 0.7 && location.Y > mDrawAreaBounds.MinY * 0.3) ||
                (location.X < mLayer.ContentSize.Width * 0.175 && location.X > 0 &&
                location.Y < mDrawAreaBounds.MinY * 0.7 && location.Y > mDrawAreaBounds.MinY * 0.3))
            {
                if (mThemeColors.IsRandomColorTheme())
                {
                    mThemeColors = new ThemeColors(mThemeColors.GetColorIndex());
                }
                else
                {
                    int numColorThemes = SharedParams.GetNumUnlockedColorThemes();
                    int index = mThemeColors.GetColorIndex();

                    index++;
                    if (ThemeColors.IsRandomColorThemeIndex(index)) { index++; } // Skip random color theme
                    if (index >= numColorThemes) { index = 0; }

                    mThemeColors = new ThemeColors(index);
                }

                mResetMineCenters = true;
                StartNewRound();
                return;
            }

            const string fixGridSize = "Grid: Max Granularity";
            const string dynamicGridSize = "Grid: Dynamic Optimized";
            const string replayLastDraw = "Replay Last Draw";
            const string showAIAngles = "Show Grid Moves";
            const string hideAIAngles = "Hide Grid Moves";
            const string mostRepetitive = "        Similar Curves";
            const string mostVaried = "       Distinct Changes";
            const string mymind = "      Last Mind Stream";
            const string highlightHabits = "Highlight Habits";
            const string highlightChanges = "Highlight Changes";
            if (mShowingSpecialDetails && mExtraInfoLabel2 != null && mExtraInfoLabel2.Parent == mLayer &&
                location.X > mDrawAreaBounds.MaxX * 0.6f && location.Y > mDrawAreaBounds.MaxY - LargeFontSize && location.Y < mDrawAreaBounds.MaxY + LargeFontSize * 2)
            {
                if (mExtraInfoLabel2 != null && mExtraInfoLabel2.Text == showAIAngles)
                {
                    DrawExtraInfoLabel(info2:hideAIAngles, clickable:true);
                    ReAddAllLinesToLayer();
                    ClearAllLines();
                    RedrawAllLines(drawAngles: true);
                    
                    if (mFixedGridGranularity) { mControlLabel.Text = fixGridSize; }
                    else { mControlLabel.Text = dynamicGridSize; }

                    DrawShowGridMovesTextandOverlay();
                }
                else if (mExtraInfoLabel2 != null && mExtraInfoLabel2.Text == hideAIAngles)
                {
                    DrawExtraInfoLabel(info2:showAIAngles, clickable:true);
                    ReAddAllLinesToLayer();
                    ClearAllLines();
                    RedrawAllLines();
                    if (mHeaderText != null) { mHeaderText.Text = ""; }
                    mControlLabel.Text = highlightHabits;

                    RemoveTextUnderlay();
                }
                return;
            }
            if (mTextUnderlay != null && mExtraInfoLabel2 != null && mExtraInfoLabel2.Text == hideAIAngles
                && mTextUnderlay.BoundingRect.ContainsPoint(location))
            {
                if (mHeaderText != null) { mHeaderText.Text = ""; }
                RemoveTextUnderlay();
                return;
            }
            if ((mPlayMode != PlayMode.Continuity || mContinuityScoredDrawStartIndex > 0 || mContinuityScoredDrawEndIndex > 0 || !IsInContinuitySeries()) && mScoredDraws.Count > 0 && location.Y < mDrawAreaBounds.MinY * 0.45 && location.X > mLayer.ContentSize.Width * 0.32f && location.X < mLayer.ContentSize.Width * 0.68f && mMovesRemaining == mMovesPerRound)
            {
                MinimizeScreenContent(v2: true);

                // Since we're showing last draw, remove the rank label if needed, and header text
                CleanRankLabelContent();
                if (mHeaderText != null) { mHeaderText.Text = ""; }

                if (!mShowingSpecialDetails || mControlLabel.Text == ShowLastDraw)
                {
                    mLineRedrawIndex = LineRedrawOff;
                    mDelayContinuityGameStart = true;
                    ReAddAllLinesToLayer();
                    ClearAllLines();
                    RedrawAllLines();
                    if (!mExperimentMode) { DrawExtraInfoLabel(mymind); }
                    mControlLabel.Text = highlightHabits;
                    mShowingSpecialDetails = true;

                    DrawExtraInfoLabel(info2:showAIAngles, clickable:true);
                }
                else if (mShowingSpecialDetails)
                {
                    if (mControlLabel.Text == fixGridSize)
                    {
                        mFixedGridGranularity = false;
                        mControlLabel.Text = dynamicGridSize;
                        MainActivity.SaveIntValue(0, FixedUpdateGranularity);
                        DrawShowGridMovesTextandOverlay();
                    }
                    else if (mControlLabel.Text == dynamicGridSize)
                    {
                        mFixedGridGranularity = true;
                        mControlLabel.Text = fixGridSize;
                        MainActivity.SaveIntValue(1, FixedUpdateGranularity);
                        DrawShowGridMovesTextandOverlay();
                    }
                    else if (mControlLabel.Text == highlightHabits)
                    {
                        mLineRedrawIndex = LineRedrawOff;
                        ReAddAllLinesToLayer();
                        ClearAllLines();
                        if (!HighlightHabits())
                        {
                            DrawExtraInfoLabel("No Habits!", sizeMult1:1.1f);
                        }
                        else
                        {
                            DrawExtraInfoLabel(mostRepetitive, sizeMult1:1.0f, textColor:ThemeColors.GetColorForTargetLuminance(mThemeColors.DistinctColorA, 0.8));
                        }

                        mControlLabel.Text = highlightChanges;
                    }
                    else if (mControlLabel.Text == highlightChanges)
                    {
                        mLineRedrawIndex = LineRedrawOff;
                        ReAddAllLinesToLayer();
                        ClearAllLines();
                        if (!HighlightHabits(true))
                        {
                            DrawExtraInfoLabel("No Changes!", sizeMult1: 1.1f);
                        }
                        else
                        {
                            DrawExtraInfoLabel(mostVaried, sizeMult1:1.0f, textColor:ThemeColors.GetColorForTargetLuminance(ThemeColors.InvertColor(mThemeColors.DistinctColorA), 0.8));
                        }
                        mControlLabel.Text = replayLastDraw;
                    }
                    else // Show Last Draw; this time, show the replay draw
                    {
                        ReAddAllLinesToLayer();
                        ClearAllLines();
                        mLineRedrawIndex = 0;
                        mControlLabel.Text = highlightHabits;
                        DrawExtraInfoLabel(mymind);
                        mControlLabel.Text = ShowLastDraw;
                    }
                }

                return;
            }

            if (location.Y >= mDrawAreaBounds.MinY && location.Y <= mDrawAreaBounds.MaxY)
            {
                const long MinDurationBetweenRound = 125 * TimeSpan.TicksPerMillisecond;
                if (DateTime.UtcNow.Ticks > mLastTouchMoveTicks + MinDurationBetweenRound)
                {
                    ReadyForDrawing();
                }
            }

            touchEvent.StopPropogation();
        }

        bool mIsFingerPausedOrLifted;
        private void HandleTouchesMoved(List<CCTouch> touches, CCEvent touchEvent)
        {
            if (touches[0].Id != 0) { return; }

            if (!mReadyForDrawing) { return; }

            if (mDelayContinuityGameStart && mPlayMode == PlayMode.Continuity) { return; }

            mIsFingerPausedOrLifted = false;

            mLastTouchMoveTicks = DateTime.UtcNow.Ticks;
            HandleTouchesMoved_Orig(touches[0].Location);

            touchEvent.StopPropogation();
        }

        // The original touch move handler, which processed raw input from the finger.
        private void HandleTouchesMoved_Orig(DrawPoint location)
        {
            bool skipGuideParamRecalc = true;
            if (mMovesRemaining == mMovesPerRound) { skipGuideParamRecalc = false; }

            if (mEndRoundEarly)
            {
                UpdateScoreCircle(false, false);
            }
            else if (mHistory != null && mMovesRemaining > 0)
            {
                bool hit = mMineManager.HitTestMine(location);
                if (mPlayMode == PlayMode.VoidStormV2)
                {
                    if (hit) { mVoidStormLevelScoreAdjuster -= VoidStormScoreBonusDecrementPerHit; }
                }
                else
                {
                    mEndRoundEarly = hit;
                    if (mEndRoundEarly) { mConseqEndRoundEarlyDueToMines++; }
                }

                DrawPoint prevLocation;
                if (!mHistory.GetLastTouchPoint(out prevLocation))
                {
                    prevLocation = location;
                }
                mHistory.Add(location);

                double distanceFromLast = UpdateMovementState();

                if (distanceFromLast >= 0)
                {
                    skipGuideParamRecalc = false;
                    UpdateScoreCircle(false);

                    if (location.X < mDrawAreaBounds.MinX || location.X > mDrawAreaBounds.MaxX ||
                        location.Y < mDrawAreaBounds.MinY || location.Y > mDrawAreaBounds.MaxY)
                    {
                        mNumOutOfBoundsMoves++;
                        DrawBoundsLine(DefaultBoundsWidth * mNumOutOfBoundsMoves * 2, 1.25);

                        if (mNumOutOfBoundsMoves > OutOfBoundsPenaltyMoves)
                        {
                            MainActivity.SoundEffects.RoundEndEarly();
                            mEndRoundEarly = true;
                        }
                    }
                    else
                    {
                        if (mNumOutOfBoundsMoves > 0)
                        {
                            DrawBoundsLine(DefaultBoundsWidth);
                        }
                        mNumOutOfBoundsMoves = 0;
                    }
                }

                DrawTraceLine(prevLocation.X, prevLocation.Y, location.X, location.Y);

                if (mMovesRemaining == 0)
                {
                    if (mFinalScore >= 1)
                    {
                        MainActivity.SoundEffects.RoundCompleteWin();
                    }
                    else
                    {
                        MainActivity.SoundEffects.RoundComplete();
                    }
                }
            }

            DrawDrawingGuide(skipGuideParamRecalc, location.X, location.Y);
        }

        private void HandleTouchesEnded(List<CCTouch> touches, CCEvent touchEvent)
        {
            if (touches[0].Id != 0) { return; }

            if (!mReadyForDrawing) { return; }

            if (mDelayContinuityGameStart) { mDelayContinuityGameStart = false; }

            if (mMovesRemaining == mMovesPerRound)
            {
                EndRound(touches[0].Location);
            }
            else if (mMovesRemaining > 0 && mMovesRemaining < mMovesPerRound &&
                !mEndRoundEarly)
            {
                // We're pausing
                mHistory.GetLastTouchPoint(out mLastTouchUpPausePoint);

                mHistory.Clear();

                mTraceLine.Clear();
                mTraceLine.Cleanup();
                mTraceLine.Visible = false;
            }
            else if (mMovesRemaining == 0)
            {
                mNumberOfFinishedPlaysThisSession++;
                EndRound(touches[0].Location);
            }
            else if (mMovesRemaining != mMovesPerRound)
            {
                DrawPoint prevCursorLocation;
                if (!mHistory.GetLastTouchPoint(out prevCursorLocation))
                {
                    EndRound(touches[0].Location);
                }
                else
                {
                    EndRound(prevCursorLocation);
                }
            }
        }

        void ApplyBonusesToFinalScore()
        {
            if (mMovesRemaining == mMovesPerRound) { return; }

            mFinalScore += mVoidStormLevelScoreAdjuster;
            mLastFinalScoreBonus = mVoidStormLevelScoreAdjuster;
        }

        const double DynamicUpdateDistanceTargetDegrees = (1.0 / (Math.PI * 1.6)) * 360; // TODO: Seems to work, but make more mathematical. 
        void UpdateDynamicUpdateDistance()
        {
            if (mMovesRemaining == 0 && mSumDegrees > 0 && !mFixedGridGranularity && !mExperimentMode)
            {
                double factor = SharedParams.UpdateDistanceAngleDeltaFactor;
                double maxDelta = SharedParams.MaxUpdateDistanceDelta;

                if (mPlayMode == PlayMode.Standard && mBestScore < 100)
                {
                    // Increase them when score hasn't yet reached 100
                    factor *= 1.5;
                    maxDelta *= 1.5;
                }

                double avgDegrees = (DynamicUpdateDistanceTargetDegrees * (1.0 - factor)) + (factor * (mSumDegrees / mMovesPerRound));

                // Update the update distance draw granularity. Cap it to avoid abrupt changes. 
                double updateDistanceDiv = avgDegrees / DynamicUpdateDistanceTargetDegrees;
                if (updateDistanceDiv > 1 + maxDelta) { updateDistanceDiv = 1 + maxDelta; }
                else if (updateDistanceDiv < 1 - maxDelta) { updateDistanceDiv = 1 - maxDelta; }

                mUpdateDistanceDrawGranularity /= updateDistanceDiv;
                MainActivity.SaveIntValue((int)(mUpdateDistanceDrawGranularity * 10000.0), CurrentUpdateGranularity);

                //System.Diagnostics.Debug.WriteLine("Avg angle: " + avgDegrees + ", Div: " + updateDistanceDiv);
            }
            mSumDegrees = 0;
        }

        void EndRound(DrawPoint guideCneter)
        {
            UpdateDynamicUpdateDistance();
            ApplyBonusesToFinalScore();
            InitGlitter(guideCneter);
            UpdateBestScores();

            if (mMovesRemaining != mMovesPerRound)
            {
                mInitializedContinuityScoredDrawStartIndex = false;
                mResetMineCenters = true;
            }
            else
            {
                mResetMineCenters = false;
            }

            StartNewRound();
        }

        void InitGlitter(DrawPoint location)
        {
            if (!mDrawingGlitter && mGlitterDots.Count > 0)
            {
                double glitterScore = mFinalScore;
                if (glitterScore < 0.99) { glitterScore = 0.99; }
                double durationMult = 1.0;

                if (glitterScore >= 1 && !mExperimentMode)
                {
                    durationMult = (glitterScore / 2) + 0.5;
                    glitterScore *= Math.Sqrt(GlitterEffect100PlusBoost);
                }

                List<GlitterDot> tempGlitterDots = new List<GlitterDot>();
                List<int> addedIndexes = new List<int>();
                for (double j = 0; j < mGlitterDots.Count; j++)
                {
                    int floorJ = Convert.ToInt32(Math.Floor(j));
                    if (!addedIndexes.Contains(floorJ))
                    {
                        tempGlitterDots.Add(mGlitterDots[floorJ]);
                        addedIndexes.Add(floorJ);
                    }
                }
                mGlitterDots.Clear();
                mGlitterDots = null;
                mGlitterDots = tempGlitterDots;

                double glitterDuration = 1000.0 * durationMult;

                long now = DateTime.UtcNow.Ticks;
                long duration = (long)((glitterDuration * 0.5) * TimeSpan.TicksPerMillisecond);
                long offset = (long)((glitterDuration * 0.5 / mGlitterDots.Count) * TimeSpan.TicksPerMillisecond);

                double radiusMult = Math.Pow(((glitterScore / 2) + 0.5), 1.2) * 1.2;

                for (int j = 0; j < mGlitterDots.Count; j++)
                {
                    long start = now + (j * offset);
                    long end = start + duration;
                    mGlitterDots[j].SetStartEndDrawTicks(start, end);
                    mGlitterDots[j].SetRadiusMultiplier(radiusMult);
                }
            }
        }

        bool IsIdleGlitterMuted()
        {
            return mExperimentMode || (mFirstPlay || mShowingSpecialDetails || (mPlayMode == PlayMode.Standard && mBestScore < 100 && mNumberOfFinishedPlaysThisSession == 0));
        }

        void ProcessIdleGlitter(/*bool cleanup = false*/)
        {
            int numIdleGlitterDots = NumIdleGlitterDots;
            float radiusMult = 1f;
            if (mReadyForDrawing || mLineRedrawIndex >= 0)
            {
                // We won't add any new glitter dots, but will still process the ones we have
                numIdleGlitterDots = 0;
            }
            else
            {
                if (IsIdleGlitterMuted())
                {
                    // These settings will be for newly added glitter dots only
                    numIdleGlitterDots = NumIdleGlitterDotsMuted;
                    radiusMult = IdleGlitterMutedRadiusMult;
                }
                else if (mPlayMode != PlayMode.Continuity)
                {
                    if (mLastFinalScore > 1 && !mExperimentMode)
                    {
                        // Bigger idle glitter for higher scores
                        radiusMult *= (GlitterEffect100PlusBoost * (float)Math.Pow(mLastFinalScore, 2));
                        numIdleGlitterDots = (int)(numIdleGlitterDots * radiusMult);
                    }
                }
                else // Continuity
                {
                    if (mContinuityLastSeriesScore > 1)
                    {
                        radiusMult *= (GlitterEffect100PlusBoost * (float)Math.Pow(mContinuityLastSeriesScore, 2));
                        numIdleGlitterDots = (int)(numIdleGlitterDots * radiusMult);
                    }
                }

                // We're in idle mode, increase the glitter size slightly 
                radiusMult *= 1.25f;
            }

            long now = DateTime.UtcNow.Ticks;
            while (mIdleGlitterDots.Count < numIdleGlitterDots)
            {
                DrawRect bounds = mDrawAreaBounds;
                if (mDetailsOverlay != null) { bounds = mLayer.BoundingBox; }

                DrawPoint pt;
                if (mShowingSpecialDetails)
                {
                    // Draw the glitter more around the edges
                    float maxDistFromEdge = (bounds.MaxX - bounds.MinX) * 0.0417f;
                    int side = mRnd.Next(0, 4);
                    if (side == 0)
                    {
                        pt = new DrawPoint(
                            RandomNumberGenerator.GetInt32((int)bounds.MinX, (int)(bounds.MinX + maxDistFromEdge)),
                            RandomNumberGenerator.GetInt32((int)bounds.MinY, (int)bounds.MaxY));
                    }
                    else if (side == 1)
                    {
                        pt = new DrawPoint(
                            RandomNumberGenerator.GetInt32((int)(bounds.MaxX - maxDistFromEdge), (int)(bounds.MaxX)),
                            RandomNumberGenerator.GetInt32((int)bounds.MinY, (int)bounds.MaxY));
                    }
                    else if (side == 2)
                    {
                        pt = new DrawPoint(
                            RandomNumberGenerator.GetInt32((int)bounds.MinX, (int)bounds.MaxX),
                            RandomNumberGenerator.GetInt32((int)bounds.MinY, (int)(bounds.MinY + maxDistFromEdge)));
                    }
                    else
                    {
                        pt = new DrawPoint(
                           RandomNumberGenerator.GetInt32((int)bounds.MinX, (int)bounds.MaxX),
                            RandomNumberGenerator.GetInt32((int)(bounds.MaxY - maxDistFromEdge), (int)bounds.MaxY));
                    }
                }
                else
                {
                    pt = new DrawPoint(
                        RandomNumberGenerator.GetInt32((int)bounds.MinX, (int)bounds.MaxX),
                        RandomNumberGenerator.GetInt32((int)bounds.MinY, (int)bounds.MaxY));
                }

                long duration = RandomNumberGenerator.GetInt32(2175, 3260) * TimeSpan.TicksPerMillisecond;
                long offsetFromNow = now + RandomNumberGenerator.GetInt32(0, 4350) * TimeSpan.TicksPerMillisecond;
                float radius = DefaultGlitterRadiius * radiusMult * (float)(mRnd.NextDouble() + (1/Math.E));

                DrawColor color = SharedParams.GetRandomIdleGlitterColor(mThemeColors, mRnd);

                GlitterDot dot = new GlitterDot(ref mLayer, pt, color, radius, 0.4, 1.6, true);
                dot.SetStartEndDrawTicks(offsetFromNow, offsetFromNow + duration);
                mIdleGlitterDots.Add(dot);
            }

            try
            {
                for (int j = 0; j < mIdleGlitterDots.Count; j++)
                {
                    if (mIdleGlitterDots[j].ShowOrHide())
                    {
                        // We've drawn, then hidden, the glitter, so we're done with it.
                        mIdleGlitterDots.RemoveAt(j);
                        j--;
                    }
                }
            }
            catch { }
        }

        const double RecentScoreGlitterThreshold = 0.0; 
        void AddIdleGlitterWhileDrawing(double recentScore, DrawPoint fingerLocation, DrawColor color)
        {
            recentScore *= (1 + mRecentAverageScore);
            if (recentScore < RecentScoreGlitterThreshold || mScoreFeedbackHidden) { return; }
            if (mExperimentMode) { recentScore *= 0.667; }

            double travelRadius = 0.75 * ((ScoreCircleRadius + (recentScore * ScoreCircleRadius)));

            DrawPoint travelTo = new DrawPoint(
                fingerLocation.X + (float)(travelRadius * (mRnd.NextDouble() - 0.5)),
                fingerLocation.Y + (float)(travelRadius * (mRnd.NextDouble() - 0.5)));

            long offsetFromNow = DateTime.UtcNow.Ticks;

            double durationMult = 1000 * Math.E;
            double sizeMult = Math.E;

            long duration = (400 + (long)(recentScore * durationMult)) * TimeSpan.TicksPerMillisecond;

            // Start 25% the way into the 'growing' phase.
            long durationAdd = duration / 8;
            duration += durationAdd;
            offsetFromNow -= durationAdd;

            float radiusMult = 0.8f;
            if (mPlayMode == PlayMode.VoidStormV2) { radiusMult *= 0.707f; }
            float radius = radiusMult * DefaultGlitterRadiius * (float)(1 + (recentScore * sizeMult));

            GlitterDot dot = new GlitterDot(ref mLayer, fingerLocation, color, radius, 0.4, 1.6, true, _travelPt:travelTo);
            dot.SetStartEndDrawTicks(offsetFromNow, offsetFromNow + duration);
            mIdleGlitterDots.Add(dot);
        }

        void Update100PlusAvgScores()
        {
            mSumScores100Plus += mFinalScore * 100;
            mConseqGamesScoring100Plus++;
            if ((mSumScores100Plus / mConseqGamesScoring100Plus) < 100)
            {
                mSumScores100Plus = 0;
                mConseqGamesScoring100Plus = 0;
            }
            MainActivity.SaveIntValue(mConseqGamesScoring100Plus, ConseqGames100Plus);
            MainActivity.SaveIntValue((int)(100 * mSumScores100Plus), SumScores100Plus);

            if (mConseqGamesScoring100Plus > mBestConseqGamesScoring100Plus)
            {
                mBestConseqGamesScoring100Plus = mConseqGamesScoring100Plus;
                MainActivity.SaveIntValue(mBestConseqGamesScoring100Plus, BestConseqGames100Plus);
            }
        }

        void UpdateBestScores()
        {
            // Only update best scores after at least 10 moves
            if (!mExperimentMode && (mMovesRemaining < (mMovesPerRound - MinMovesToTriggerEnd)))
            {
                if (mFinalScore * 100 > MinParticipationScore && !mReachedParticipationScore)
                {
                    MainActivity.SaveIntValue(1, ReachedParticipationThreshold);
                    mReachedParticipationScore = true;
                }
                if (mFinalScore * 100 > MinScoreForTutorialFinish && !mReachedTutorialFinishScore)
                {
                    MainActivity.SaveIntValue(1, ReachedNormalScoreCircleSizeThreshold);
                    mReachedTutorialFinishScore = true;
                }

                bool updateTitleUnlockCount = false;

                if (mPlayMode == PlayMode.Continuity)
                {
                    mContinuitySeriesScores.Add(mFinalScore);

                    Update100PlusAvgScores();

                    if (mContinuitySeriesScores.Count >= mContinuityGamesCount)
                    {
                        double seriesAverage = mContinuitySeriesScores.Average();
                        mContinuityLastSeriesScore = seriesAverage;

                        if (mLongContinuitySeries)
                        {
                            double oldBestScore = mContinuityLongSeriesBestScore;

                            if (seriesAverage * 100 > mContinuityLongSeriesBestScore)
                            {
                                mContinuityLongSeriesBestScore = seriesAverage * 100;
                                MainActivity.SaveIntValue((int)(100 * mContinuityLongSeriesBestScore), SharedParams.ContinuityBestScoreLongSeries);
                            }

                            else if (oldBestScore < SharedParams.Continuity_3Score && mContinuityLongSeriesBestScore >= SharedParams.Continuity_3Score)
                            {
                                updateTitleUnlockCount = true;
                            }
                            else if (oldBestScore < SharedParams.Continuity_4Score && mContinuityLongSeriesBestScore >= SharedParams.Continuity_4Score)
                            {
                                updateTitleUnlockCount = true;
                            }
                        }
                        else
                        {
                            double oldBestScore = mContinuitySeriesBestScore;

                            if (seriesAverage * 100 > mContinuitySeriesBestScore)
                            {
                                mContinuitySeriesBestScore = seriesAverage * 100;
                                MainActivity.SaveIntValue((int)(100 * mContinuitySeriesBestScore), SharedParams.ContinuityBestScore);
                            }

                            if (oldBestScore < SharedParams.Continuity_1Score && mContinuitySeriesBestScore >= SharedParams.Continuity_1Score)
                            {
                                updateTitleUnlockCount = true;
                            }
                            else if (oldBestScore < SharedParams.Continuity_2Score && mContinuitySeriesBestScore >= SharedParams.Continuity_2Score)
                            {
                                updateTitleUnlockCount = true;
                            }
                        }

                        mRedrawRankLabel = true;
                        mIsContinuityStarted = false;
                    }
                }
                else if (mPlayMode == PlayMode.VoidStormV2)
                {
                    double oldBestScore = mVoidStormBestScore;

                    bool immediateBestScore = false;

                    if (mFinalScore * 100 > mVoidStormBestScore)
                    {
                        immediateBestScore = true;
                        mVoidStormBestScore = mFinalScore * 100;
                        MainActivity.SaveIntValue((int)(100 * mVoidStormBestScore), SharedParams.VoidStormBestScore);
                    }

                    // Check for new achievement.
                    // Note: We do 'if' rather than 'else if' so we don't skip over achivements.
                    bool newAch = false;
                    if (oldBestScore < SharedParams.RankVoid_2Score && mVoidStormBestScore >= SharedParams.RankVoid_2Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }
                    if (oldBestScore < SharedParams.RankVoid_3Score && mVoidStormBestScore >= SharedParams.RankVoid_3Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }
                    if (oldBestScore < SharedParams.RankVoid_4Score && mVoidStormBestScore >= SharedParams.RankVoid_4Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }
                    if (oldBestScore < SharedParams.RankVoid_5Score && mVoidStormBestScore >= SharedParams.RankVoid_5Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }

                    if (newAch)
                    {
                        mRedrawRankLabel = true;
                    }
                    if (immediateBestScore)
                    {
                        SetMaxSubModeState();
                    }
                }
                else // Basic Game
                {
                    double oldBestScore = mBestScore;
                    double oldBestTargetAnglesScore = mBestLongGameScore;

                    bool immediateBestScore = false;

                    if (mLongGameEnabled)
                    {
                        if (mFinalScore * 100 > mBestLongGameScore)
                        {
                            immediateBestScore = true;
                            mBestLongGameScore = mFinalScore * 100;
                            MainActivity.SaveIntValue((int)(100 * mBestLongGameScore), SharedParams.BestTargetAnglesScore);
                        }
                    }
                    else 
                    {
                        if (mFinalScore * 100 > mBestScore)
                        {
                            immediateBestScore = true;
                            mBestScore = mFinalScore * 100;
                            MainActivity.SaveIntValue((int)(100 * mBestScore), SharedParams.StandardBestScore);
                        }
                    }
                    Update100PlusAvgScores();

                    // Check for new achievement.
                    // Note: We do 'if' rather than 'else if' so we don't skip over achivements.
                    bool newAch = false;
                    if (oldBestScore < SharedParams.RankBasic_1Score && mBestScore >= SharedParams.RankBasic_1Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }
                    if ((oldBestScore < SharedParams.RankBasic_2Score) && mBestScore >= SharedParams.RankBasic_2Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }
                    if ((oldBestScore < SharedParams.RankBasic_3Score) && mBestScore >= SharedParams.RankBasic_3Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }
                    if ((oldBestScore < SharedParams.ContinuityUnlock) && mBestScore >= SharedParams.ContinuityUnlock)
                    {
                        updateTitleUnlockCount = true;
                    }
                    if (mLongGameEnabled && (oldBestTargetAnglesScore < SharedParams.RankBasic_4Score) && mBestLongGameScore >= SharedParams.RankBasic_4Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }
                    if (mLongGameEnabled &&  (oldBestTargetAnglesScore < SharedParams.RankBasic_5Score) && mBestLongGameScore >= SharedParams.RankBasic_5Score)
                    {
                        newAch = true;
                        updateTitleUnlockCount = true;
                    }

                    if (newAch)
                    {
                        mRedrawRankLabel = true;
                    }
                }

                if (updateTitleUnlockCount)
                {
                    SharedParams.UpdateUnlocks();
                }
            }
        }

        double UpdateMovementState()
        {
            double ret = -1;

            if (mMovesRemaining <= 0 || mEndRoundEarly) { return ret; }

            double distanceFromLast = mHistory.GetAggregateDistance();
            if (distanceFromLast >= mUpdateDistance)
            {
                // Get the direction state for this tile, and record it
                double absState;
                double degrees = 0;
                if (!mHistory.GetDirectionState(out absState, out degrees, updateDistance: mUpdateDistance))
                {
                    return 0;
                }

                double state = SharedParams.VirtualizeState(absState, ref mLastState, ref mLastState2, ref mLastStateVirt, ref mLatestRelativeAngle, useVirtualization:!mNoVirtualization);  

                ret = distanceFromLast;
                mHistory.ReduceAggregateDistance(mUpdateDistance, mUpdateDistance);

                mSumDegrees += mSnapGridMode ? mLatestRelativeAngle : (degrees > 180 ? 180 : degrees);

                mMovesRemaining--;
                int moves = (mMovesPerRound - mMovesRemaining);

                mEngine.RecordMoveAndGetPredicted(state, out Move exp);
                double score = mEngine.GetScoringWeight(state, exp, out double add, out double neg);
                mPositiveTotal += add;
                mNegativeTotal += neg;
                mLastScore = score;

                if (mPositiveTotal > 0 || mNegativeTotal > 0)
                {
                    if (mPositiveTotal < mNegativeTotal)
                    {
                        mFinalScore = (mPositiveTotal / mNegativeTotal);
                    }
                    else
                    {
                        mFinalScore = 2 - (mNegativeTotal / mPositiveTotal);
                    }
                    //mFinalScore = (2 * mPositiveTotal) / (mPositiveTotal + mNegativeTotal);
                    mFinalScore *= ((double)moves / (double)mMovesPerRound);;
                }

                // Update the recent average score.
                mRecentScores.Add(mLastScore);
                mRecentScores.RemoveAt(0);
                mRecentAverageScore = mRecentScores.Average();

                UpdateScoreLabel();

                // Spawn mines and orbs
                if (mPlayMode == PlayMode.VoidStormV2)
                {
                    int minesToSpawn = MinesToSpawn();
                    int minesToRemove = MinesToRemove();

                    for (int j = 0; j < minesToSpawn; j++)
                    {
                        double radiusMult = 1 + ((double)(mMovesPerRound - mMovesRemaining) * (VoidStormBaseMineAreaIncrease) / (double)mMovesPerRound);
                        mMineManager.SetMineRadiusMult(Math.Sqrt(radiusMult));

                        DrawPoint pt;
                        mHistory.GetLastTouchPoint(out pt);
                        mMineManager.AddRandomMine(pt);
                    }
                    for (int j = 0; j < minesToRemove; j++)
                    {
                        mMineManager.RemoveRandomMines(1, skipLastAdded: true);
                    }
                }
                else // Standard or Continuity 
                {
                    bool spawn = false;
                    int currentMineCount = mMineManager.GetCurrentMineCount();

                    if (currentMineCount < (mLongGameEnabled ? StandardMineCountLongGame : StandardMineCount))
                    {
                        if (currentMineCount == 0 || // Spawn the first mine right away
                            (mMovesPerRound - mMovesRemaining) % (mLongGameEnabled ? StandardMineInitialSpawnMovesLongGame : StandardMineInitialSpawnMoves) == 0)
                        {
                            spawn = true;
                        }
                    }
                    else
                    {
                        mStandardMovesSinceRespawn++;
                        if (mStandardMovesSinceRespawn >= StandardMineRespawnMoves)
                        {
                            mStandardMovesSinceRespawn = 0;
                            spawn = true;
                        }
                    }

                    if (spawn)
                    {
                        // We don't start spawning mines until we have NumScoresToAverage moves
                        DrawPoint pt;
                        mHistory.GetLastTouchPoint(out pt);
                        if (currentMineCount >= (mLongGameEnabled ? StandardMineCountLongGame : StandardMineCount))
                        {
                            mMineManager.RemoveRandomMines(1); // During random selection of mine to remove, skip an active mine once
                        }
                        mMineManager.AddRandomMine(pt);
                    }
                }
            }

            return ret;
        }

        public override void Update(float dt)
        {
            if (mUpdateThreadLocked) { return; }

            // Active drawing functions
            if (mReadyForDrawing &&
                DateTime.UtcNow.Ticks > (mTicksLastEmergeDecayDraw + (EmergDecayDrawPeriod * TimeSpan.TicksPerMillisecond)))
            {
                mTicksLastEmergeDecayDraw = DateTime.UtcNow.Ticks;
                mMineManager.DrawEmergingAndDecayingMines();
            }

            UpdateTimeRemaining();

            // Idle functions
            try
            {
                if (mDrawingGlitter && mGlitterDots.Count > 0)
                {
                    for (int j = 0; j < mGlitterDots.Count; j++)
                    {
                        if (mGlitterDots[j].ShowOrHide())
                        {
                            // We've drawn, then hidden, the glitter, so we're done with it.
                            mGlitterDots[j].Clear();
                            mGlitterDots.RemoveAt(j);
                            j--;
                        }
                    }
                }

                bool firstRankRedraw = false;
                if (mGlitterDots.Count == 0)
                {
                    if (mDrawingGlitter)
                    {
                        mPulsate.ResetScaleFactor();
                        mDrawingGlitter = false;
                        if (mRedrawRankLabel) { firstRankRedraw = true; }
                    }
                }

                if (!mReadyForDrawing && !mDrawingGlitter && mDetailsOverlay == null &&
                    DateTime.UtcNow.Ticks > (mTicksLastPulsate + (SharedParams.PulsatePeriod * TimeSpan.TicksPerMillisecond)))
                {
                    mTicksLastPulsate = DateTime.UtcNow.Ticks;

                    mPulsate.NextScaleFactor();

                    UpdateScoreCircle(false, true);
                    if (!mShowingSpecialDetails)
                    {
                        UpdateGameModeCircle(false, true);
                        if (mPlayMode != PlayMode.Continuity) { DrawColorPicker(true); }
                    }
                    DrawBoundsLine(DefaultBoundsWidth, pulsateMode: true);
                    DrawRankLabel(!firstRankRedraw);

                    if (mShowingSpecialDetails) { DrawNextLine(); }
                }

                ProcessIdleGlitter();
            }
            catch { }
        }

        float mGuideDotRadius;
        float mRingWidthInner;
        const double ConnectingRingWidthMult = 0.1333;
        void DrawDrawingGuide(bool skipRecalParams, float x, float y)
        {
            if (mGuide == null) { return; }

            const int numDots = 8;
            float radius = GuideRadius;
            float dotRadius = GuideLargeDotRadius;

            double radianStep = (2 * Math.PI) / (double)numDots;

            float ringRadius = radius;
            if (!skipRecalParams)
            {
                mGuideDotRadius = dotRadius;
                mRingWidthInner = (int)Math.Ceiling(mGuideDotRadius * ConnectingRingWidthMult);
            }

            mGuide.Clear();
            mGuide.Cleanup();
            mGuide.Visible = false;

            mGuideRing.Clear();
            mGuideRing.Cleanup();
            mGuideRing.Visible = false;

            mGuideRing.ZOrder = mLayer.ChildrenCount; // Keep on top, but under main guide
            mGuide.ZOrder = mLayer.ChildrenCount + 1; // Keep guide on top

            // Set the dot color based on the line coloring (relative angle). Heuristics to keep it from feeling jarring. 
            DrawColor dotColor = mScoredDraws.Count > 0 ? mScoredDraws[mScoredDraws.Count - 1].color : mGuideBaseColor;
            if (!mSnapGridMode && mScoredDraws.Count >= 2)
            {
                // Give the color changes a smoother feel
                dotColor = ThemeColors.AverageColors(dotColor, mScoredDraws[mScoredDraws.Count - 2].color);
            }

            if (mSnapGridMode)
            {
                float radius2 = radius + GuideLargeDotRadius;
                DrawPoint ptOrigin = mHistory.GetSnapGridOriginPoint();

                double colorWeight = mHistory.GetAggregateDistance() / mUpdateDistance;
                DrawColor color1 = ThemeColors.AverageColorsWeighted(mThemeColors.BackgroundColor, colorWeight, dotColor);
                DrawColor color2 = ThemeColors.AverageColorsWeighted(mThemeColors.BackgroundColor, 1 - colorWeight, dotColor);

                if (mMovesRemaining > 0 && !mEndRoundEarly)
                {
                    DrawPoint center = new DrawPoint(x, y);
                    // Draw ring in center of guide
                    radius2 -= 2 * (GuideLargeDotRadius * (float)colorWeight);
                    mGuideRing.DrawCircle(center, radius2, mGuideBaseColor);
                    for (int j = 1; j < mRingWidthInner; j++)
                    {
                        mGuideRing.DrawCircle(center, radius2 - j, mGuideBaseColor);
                        mGuideRing.DrawCircle(center, radius2 + j, mGuideBaseColor);
                    }
                }

                for (int j = 0; j < numDots; j++)
                {
                    float x_ = x + (radius * (((float)Math.Cos(((double)j * radianStep)))));
                    float y_ = y + (radius * (((float)Math.Sin(((double)j * radianStep)))));
                    DrawPoint pt = new DrawPoint(x_, y_);
                    mGuide.DrawSolidCircle(pt, dotRadius, color2);
                }

                mGuideRing.DrawCircle(ptOrigin, radius2, mGuideBaseColor);
                for (int j = 1; j < mRingWidthInner; j++)
                {
                    mGuideRing.DrawCircle(ptOrigin, radius2 - j, mGuideBaseColor);
                    mGuideRing.DrawCircle(ptOrigin, radius2 + j, mGuideBaseColor);
                }

                for (int j = 0; j < numDots; j++)
                {
                    float x_ = ptOrigin.X + (radius * (((float)Math.Cos(((double)j * radianStep)))));
                    float y_ = ptOrigin.Y + (radius * (((float)Math.Sin(((double)j * radianStep)))));
                    DrawPoint pt = new DrawPoint(x_, y_);
                    mGuide.DrawSolidCircle(pt, dotRadius, color1);
                }
            }
            else
            {
                if (mMovesRemaining > 0 && !mEndRoundEarly)
                {
                    DrawPoint center = new DrawPoint(x, y);
                    // Draw ring in center of guide
                    mGuideRing.DrawCircle(center, radius, mGuideBaseColor);
                    for (int j = 1; j < mRingWidthInner; j++)
                    {
                        mGuideRing.DrawCircle(center, radius - j, mGuideBaseColor);
                        mGuideRing.DrawCircle(center, radius + j, mGuideBaseColor);
                    }
                }

                for (int j = 0; j < numDots; j++)
                {
                    float x_ = x + (radius * (((float)Math.Cos(((double)j * radianStep)))));
                    float y_ = y + (radius * (((float)Math.Sin(((double)j * radianStep)))));
                    DrawPoint pt = new DrawPoint(x_, y_);
                    mGuide.DrawSolidCircle(pt, dotRadius, dotColor);
                }
            }

            mGuide.Visible = true;
            mGuideRing.Visible = true;
        }

        void DrawTraceLine(float x1, float y1, float x2, float y2)
        {
            float traceLineWidth = TraceLineWidth;
            List<DrawPoint> lastStatePath = mHistory.GetLastDirStatePoints();

            if (!mSnapGridMode)
            {
                if (mTraceLine == null) { return; }

                if (lastStatePath != null)
                {
                    // We will draw the path since the last state update
                    mTraceLine.Clear();
                    mTraceLine.Cleanup();
                    mTraceLine.Visible = false;
                }

                DrawColor lineColor = mThemeColors.AveragedColor;
                mTraceLine.DrawLine(
                    from: new DrawPoint(x1, y1),
                    to: new DrawPoint(x2, y2),
                    color: lineColor,
                    lineWidth: traceLineWidth * 0.75f,
                    lineCap: CCLineCap.Round);
                mTraceLine.Visible = true;
            }

            if (lastStatePath != null)
            {
                double score = mLastScore;
                double lineColorMultScore = 0;
                if (!mScoreFeedbackHidden)
                {
                    lineColorMultScore = score;
                    if (mScoredDraws.Count > 2)
                    {
                        lineColorMultScore += mScoredDraws[mScoredDraws.Count - 1].score;
                        lineColorMultScore += mScoredDraws[mScoredDraws.Count - 2].score;
                        lineColorMultScore += mScoredDraws[mScoredDraws.Count - 3].score;
                    }
                    lineColorMultScore /= 4;
                }

                // Make the angles a weighted moving average.
                double relativeAngle = (mLatestRelativeAngle / 180.0);
                List<double> latestAvgRelAngle = new List<double>();
                if (mExperimentMode)
                {
                    latestAvgRelAngle.Add(relativeAngle);
                }
                else
                {
                    latestAvgRelAngle.Add(relativeAngle);
                    if (mScoredDraws.Count > 2)
                    {
                        latestAvgRelAngle.Add(mScoredDraws[mScoredDraws.Count - 1].relativeAngleRaw);
                        latestAvgRelAngle.Add(mScoredDraws[mScoredDraws.Count - 2].relativeAngleRaw);
                        latestAvgRelAngle.Add(mScoredDraws[mScoredDraws.Count - 3].relativeAngleRaw);
                    }
                    else
                    {
                        latestAvgRelAngle.Add(0.5);
                        latestAvgRelAngle.Add(0.5);
                        latestAvgRelAngle.Add(0.5);
                    }
                }
                double latestAvgRelativeAngle = latestAvgRelAngle.Average();

                // Scale the line color based on score.
                double factor = 0.4;
                if (lineColorMultScore < 0) { factor *= 2.25; } 
                double colorMult = 1.0 + (lineColorMultScore * factor);
                if (colorMult < 0.1) { colorMult = 0.1; }

                DrawColor color = SharedParams.GetColorPatternsColor(latestAvgRelativeAngle, mThemeColors);

                double secondaryColorMult = 1.0 + (lineColorMultScore * (5.0 / 7.0));
                DrawColor graphLineColor = ThemeColors.AdjustColorsForMultipler(secondaryColorMult, mThemeColors.AveragedColor);

                // Scale the line width as well
                traceLineWidth *= (float)(1.0 + (score * 0.2));

                mScoredDraws.Add(new ScoredDraw(ref mLayer, lastStatePath, color, colorMult, score, traceLineWidth, latestAvgRelativeAngle, relativeAngle, graphLineColor)); 
                mScoredDraws[mScoredDraws.Count - 1].Draw();
                AddIdleGlitterWhileDrawing(lineColorMultScore, lastStatePath[0], mScoredDraws[mScoredDraws.Count - 1].color);

                ShowDrawHistory();

                if (!mDrawingGlitter)
                {
                    float scoreRadiusMult = 0.5f;
                    DrawColor glitterCol = mThemeColors.Color2;
                    double glitterColMult;
                    glitterCol = color;
                    glitterColMult = colorMult;

                    glitterCol = ThemeColors.AdjustColorsForMultipler(glitterColMult, glitterCol.R, glitterCol.G, glitterCol.B);

                    float radius = DefaultGlitterRadiius * (float)(1.0 + (score * scoreRadiusMult));
                    if (radius < MinGlitterRadius) { radius = MinGlitterRadius; }

                    // Add to the beginning, so we draw the glitter in reverse order. 
                    mGlitterDots.Insert(0, new GlitterDot(ref mLayer, lastStatePath[lastStatePath.Count - 1], glitterCol, radius, _travelPt: lastStatePath[0]));
                }
            }
        }

        void ReAddAllIdleGlitterToLayer()
        {
            for (int j = 0; j < mIdleGlitterDots.Count; j++)
            {
                try
                {
                    mIdleGlitterDots[j].ReAddToLayer();
                }
                catch { }
            }
        }

        void PauseIdleGlitterDrawing()
        {
            for (int j = 0; j < mIdleGlitterDots.Count; j++)
            {
                try
                {
                    mIdleGlitterDots[j].SetTerminateIfDrawNotStarted();
                }
                catch { }
            }
        }

        void ReAddAllLinesToLayer(int startIndex = 0, int endIndex = 0)
        {
            if (endIndex == 0) { endIndex = mScoredDraws.Count; }
            for (int j = startIndex; j < endIndex; j++)
            {
                mScoredDraws[j].ReAddToLayer();
            }
        }

        void DrawLinesWithAngles(int start, int end)
        {
            float angleWidth = TraceLineWidth * 0.75f;
            DrawColor color = ThemeColors.AdjustColorsForMultipler(1.2, mThemeColors.AveragedColor);
            for (int j = start; j < end; j++)
            {
                mScoredDraws[j].Draw(true, color, angleWidth);
            }
        }

        void RedrawAllLines(int startIndex = 0, bool drawAngles = false, int endIndex = 0)
        {
            if (drawAngles)
            {
                DrawLinesWithAngles(startIndex, mScoredDraws.Count);
            }
            else
            {
                if (endIndex == 0) { endIndex = mScoredDraws.Count; }
                for (int j = startIndex; j < endIndex; j++)
                {
                    mScoredDraws[j].Draw();
                }
            }

            ShowDrawHistory(mPlayMode != PlayMode.Continuity || mContinuityScoredDrawStartIndex > 0 || mContinuityScoredDrawEndIndex > 0 || !IsInContinuitySeries(), startIndex, endIndex);
        }

        void ClearAllLines(bool delete = false)
        {
            foreach (ScoredDraw sc in mScoredDraws)
            {
                sc.MakeInvisible(delete);
            }
        }

        const int LineRedrawOff = -1;
        int mLineRedrawIndex = LineRedrawOff;
        const long TicksToRedrawOneLine = 70 * TimeSpan.TicksPerMillisecond; 
        long mLastRedrawLineTicks = 0;
        void DrawNextLine()
        {
            if (mLineRedrawIndex < 0 || mScoredDraws == null || mScoredDraws.Count == 0) { return; }

            long now = DateTime.UtcNow.Ticks;
            if (now > mLastRedrawLineTicks + TicksToRedrawOneLine)
            {
                mLastRedrawLineTicks = now - (now - (mLastRedrawLineTicks + TicksToRedrawOneLine));
                if (mLastRedrawLineTicks < now - TicksToRedrawOneLine || mLastRedrawLineTicks > now)
                {
                    mLastRedrawLineTicks = now;
                }

                if (mLineRedrawIndex >= 0 && mLineRedrawIndex < mScoredDraws.Count)
                {
                    mScoredDraws[mLineRedrawIndex].Draw();
                    ShowDrawHistory(true, 0, mLineRedrawIndex);
                    mLineRedrawIndex++;
                }
                else
                {
                    mLineRedrawIndex = LineRedrawOff;
                    ClearAllLines();
                    RedrawAllLines();
                    ShowDrawHistory(true, 0, mScoredDraws.Count - 1);
                }
            }
        }

        DrawNode mDrawHistoryVisual;
        DrawNode mDrawHistoryVisualColoredLine;
        // NOTE: We assume that the indexes of highlightFactors match the range of start to end. We will only use them in replayMode.
        void ShowDrawHistory(bool replayMode = false, int start = 0, int end = 0, List<double> highlightFactors = null, DrawColor? highlightColor = null, DrawColor? baseColor = null, DrawColor? highlightColor2 = null, int startHighlight2 = -1)
        {
            if (mDrawHistoryVisual == null)
            {
                mDrawHistoryVisual = new DrawNode();
                mLayer.AddChild(mDrawHistoryVisual);
                mDrawHistoryVisualColoredLine = new DrawNode();
                mLayer.AddChild(mDrawHistoryVisualColoredLine);
            }
            else
            {
                mDrawHistoryVisual.Clear();
                mDrawHistoryVisual.Cleanup();
                mDrawHistoryVisual.Visible = false;

                mDrawHistoryVisualColoredLine.Clear();
                mDrawHistoryVisualColoredLine.Cleanup();
                mDrawHistoryVisualColoredLine.Visible = false;
            }

            float startX = mLayer.ContentSize.Width * 0.01f;
            float endX = mLayer.ContentSize.Width * 0.99f;
            float endXBar = endX;
            float x = startX;
            float maxY = mDrawAreaBounds.MinY * 0.9f;
            float minY = mDrawAreaBounds.MinY * 0.1f;
            float xIncrement;

            int startIndex;
            int endIndex;
            List<int> omitIndexes = null;
            if (replayMode)
            {
                minY += (maxY - minY) * 0.4f;
                maxY += mDrawAreaBounds.MinY * 0.05f;

                int movesPerRound = mMovesPerRound + mContinuityScoredDrawFirstMoveThisGame; // BUG: Not quite right, uses the prior game's number.
                int xIncDenom = (end - start);
                if (xIncDenom < HistoryScrollPercent * movesPerRound) { xIncDenom = (int)(HistoryScrollPercent * movesPerRound); }
                xIncrement = (endX - startX) / xIncDenom;

                startIndex = start;
                endIndex = end;
            }
            else
            {
                xIncrement = GetDrawHistoryXIncrementDenominator(startX, endX);

                startIndex = 0;
                endIndex = mScoredDraws.Count;
            }

            // Draw upper and lower bounds for graph (TODO: Do this just once, not every move)
            mDrawHistoryVisual.DrawLine(
                from: new DrawPoint(startX, maxY),
                to: new DrawPoint(endXBar, maxY),
                color: mThemeColors.AveragedColor12,
                lineWidth: 1,
                lineCap: CCLineCap.Butt);
            mDrawHistoryVisual.DrawLine(
                from: new DrawPoint(startX, minY),
                to: new DrawPoint(endXBar, minY),
                color: mThemeColors.AveragedColor12,
                lineWidth: 1,
                lineCap: CCLineCap.Butt);

            float lineWidthInner = TraceLineWidth * 1.4f; // TODO: Member
            if (replayMode) { lineWidthInner *= 0.875f; }
            float lineWidthOuter = lineWidthInner * 1.667f;

            float lastY = minY + ((maxY - minY) * 0.5f);
            float rangeHeight = ((maxY - minY) / 2);
            float midY = rangeHeight + minY;

            for (int j = startIndex; j < endIndex; j++)
            {
                if (j >= mScoredDraws.Count) { break; }

                if (omitIndexes != null && omitIndexes.Contains(j)) { continue; }

                float y = (float)(minY + ((maxY - minY) * (1 - mScoredDraws[j].relativeAngleAvg)));

                if (replayMode && highlightFactors != null)
                {
                    DrawColor hCol = highlightColor.Value;
                    if (highlightColor2 != null && j >= startHighlight2)
                    {
                        hCol = highlightColor2.Value;
                    }

                    DrawColor color = ThemeColors.AdjustColorsForMultipler(
                        ScoredDraw.GetColorMultForHighlightFactor(highlightFactors[j]),
                        ThemeColors.AverageColorsWeighted(hCol, highlightFactors[j], baseColor.Value));

                    mDrawHistoryVisual.DrawLine(
                        from: new DrawPoint(x, lastY),
                        to: new DrawPoint(x + xIncrement, y),
                        color: color,
                        lineWidth: lineWidthInner,
                        lineCap: CCLineCap.Round);
                }
                else
                {
                    mDrawHistoryVisual.DrawLine(
                        from: new DrawPoint(x, midY),
                        to: new DrawPoint(x + xIncrement, midY),
                        color: mScoredDraws[j].color,
                        lineWidth: rangeHeight,
                        lineCap: CCLineCap.Butt);

                    mDrawHistoryVisual.DrawLine(
                        from: new DrawPoint(x, lastY),
                        to: new DrawPoint(x + xIncrement, y),
                        color: mThemeColors.BackgroundColor,
                        lineWidth: lineWidthOuter,
                        lineCap: CCLineCap.Round);

                    mDrawHistoryVisualColoredLine.DrawLine(
                        from: new DrawPoint(x, lastY),
                        to: new DrawPoint(x + xIncrement, y),
                        color: mScoredDraws[j].secondaryColor,
                        lineWidth: lineWidthInner,
                        lineCap: CCLineCap.Round);
                }

                x += xIncrement;
                lastY = y;
            }

            mDrawHistoryVisual.Visible = true;
            mDrawHistoryVisualColoredLine.Visible = true;
        }

        float GetDrawHistoryXIncrementDenominator(float startX, float endX)
        {
            int depth;
            int movesPerRound = mMovesPerRound + mContinuityScoredDrawFirstMoveThisGame;
            if (mScoredDraws.Count < movesPerRound * HistoryScrollPercent)
            {
                depth = (int)(movesPerRound * HistoryScrollPercent);
            }
            else
            {
                depth = mScoredDraws.Count;
            }
            return (endX - startX) / depth;
        }

        void GetHabitHighlitingColors(out DrawColor highlight, out DrawColor baseColor)
        {
            baseColor = mThemeColors.AveragedColor;
            highlight = ThemeColors.GetColorForTargetLuminance(mThemeColors.DistinctColorA, 0.5, false); 
        }

        List<int> GetWorstScoringMoveSequence(int conseqCount, bool inverse, out double avgScore)
        {
            List<int> moves = new List<int>();
            int start = (int)((mScoredDraws.Count * ScoringMoveSequenceSearchStart) - conseqCount); // Start searching such that at least a scroll history appears
            if (start < 0) { start = 0; }
            if (mPlayMode == PlayMode.Continuity) 
            { 
                // But for Endless Change, advance the start moves 
                start += mContinuityScoredDrawFirstMoveThisGame; 
            }

            double maxScore = 0;
            int maxScoreIndex = 0;
            bool itered = false;
            for (int j = start; j < mScoredDraws.Count - (conseqCount - 1); j++)
            {
                itered = true;
                double score = 0;
                for (int i = j; i < j + conseqCount; i++)
                {
                    score += mScoredDraws[i].score;
                }
                score /= (double)conseqCount;

                if ((inverse && score > maxScore) ||
                    (!inverse && score < maxScore))
                {
                    maxScore = score;
                    maxScoreIndex = j;
                }
            }

            // Remove moves that score inverse what we're searching for
            if (itered)
            {
                avgScore = 0;
                for (int j = maxScoreIndex; j < maxScoreIndex + conseqCount; j++)
                {
                    if ((inverse && mScoredDraws[j].score > 0) ||
                        (!inverse && mScoredDraws[j].score < 0))
                    {
                        moves.Add(j);
                        avgScore += mScoredDraws[j].score;
                    }
                }
                if (moves.Count > 0)
                {
                    avgScore /= moves.Count;
                    avgScore = Math.Abs(avgScore);
                }
            }
            else
            {
                avgScore = 0;
            }

            return moves;
        }

        const double VirtualMaxHighlightFactor = 0.9;
        List<int> GetScoredMovesOfInterest(bool inverse)
        {
            int MoveCount = mExperimentMode ? ShortHighlightHabitsMoveSequence : HighlightHabitsMoveSequence;
            int MinMoves = MoveCount / 2;
            const double ScoreThresholdMax = VirtualMaxHighlightFactor * 0.8; // TODO: Check how many scored moves we get in the debugger. 
            const double ScoreThresholdDec = ScoreThresholdMax / 30.0;
            double scoreThreshold = ScoreThresholdMax;

            List<int> movesOfInterest = null;

            for (int moves = MoveCount; moves >= MinMoves; moves--)
            {
                double avgScore;
                movesOfInterest = GetWorstScoringMoveSequence(moves, inverse, out avgScore);

                //System.Diagnostics.Debug.WriteLine("Highlighted habit moves: " + movesOfInterest.Count + ", Score threshold: " + scoreThreshold + ", Avg: " + avgScore);
                if (avgScore >= scoreThreshold ||
                    (movesOfInterest != null && movesOfInterest.Count <= MinMoves))
                {
                    break;
                }

                scoreThreshold -= ScoreThresholdDec;
            }

            if (movesOfInterest.Count == 0) { movesOfInterest = null; }
            return movesOfInterest;
        }

        List<double> GetHighlightFactors(List<double> contWeights, List<int> movesOfInterest, int contMovingAvgCount, int scoredMovingAvgCount, bool inverse)
        {
            // Take the moving average of the contWeights, and scores, separately, using different criteria.
            // For contributing moves, we average forward. 
            List<double> contWeightsMovAvg = Enumerable.Repeat((double)0, mScoredDraws.Count).ToList();
            for (int j = 0; j < contWeights.Count; j++)
            {
                double avgWeight = 0;
                for (int i = j; i < contMovingAvgCount + j; i++)
                {
                    if (i >= contWeights.Count) { break; }

                    avgWeight += contWeights[i];
                }
                avgWeight /= contMovingAvgCount;
                contWeightsMovAvg[j] = avgWeight;
            }
            // For scored moves, we average backwards (like while playing the game). 
            List<double> scoresMovAvg = Enumerable.Repeat((double)0, mScoredDraws.Count).ToList();
            for (int j = 0; j < mScoredDraws.Count; j++)
            {
                double avgScore = 0;
                for (int i = j; i > j - scoredMovingAvgCount; i--)
                {
                    if (i < 0) { break; }

                    if (movesOfInterest.Contains(i))
                    {
                        avgScore += mScoredDraws[i].score;
                    }
                }
                avgScore /= scoredMovingAvgCount;
                scoresMovAvg[j] = avgScore;
            }

            for (int j = movesOfInterest.Min(); j < contWeightsMovAvg.Count; j++)
            { 
                contWeightsMovAvg[j] = 0;
            }

            // Calculate highlight factors based on the scores and contributing weights.
            // Also no 'inverse' factors < 0.
            List<double> highlightFactors = new List<double>();
            for (int j = 0; j < contWeightsMovAvg.Count; j++)
            {
                if (inverse)
                {
                    highlightFactors.Add(Math.Max(scoresMovAvg[j], contWeightsMovAvg[j]));
                    if (highlightFactors[j] < 0) { highlightFactors[j] = 0; }
                }
                else
                {
                    highlightFactors.Add(Math.Min(scoresMovAvg[j], contWeightsMovAvg[j]));
                    if (highlightFactors[j] > 0) { highlightFactors[j] = 0; }
                }
                highlightFactors[j] = Math.Abs(highlightFactors[j]);

                highlightFactors[j] /= VirtualMaxHighlightFactor;

                // Make sure factors are in range 0 to 1.
                if (highlightFactors[j] < 0) { highlightFactors[j] = 0; }
                else if (highlightFactors[j] > 1) { highlightFactors[j] = 1; }
            }

            return highlightFactors;
        }

        bool HighlightHabits(bool inverse = false)
        {
            List<int> movesOfInterest = GetScoredMovesOfInterest(inverse);
            if (movesOfInterest == null || movesOfInterest.Count == 0)  { return false; }

            // Get the total contributing weight for every move
            List<double> contWeights = Enumerable.Repeat((double)0, mScoredDraws.Count).ToList();
            List<double> temp = mEngine.GetContributingWeights(movesOfInterest);
            for (int i = 0; i < contWeights.Count; i++)
            {
                contWeights[i] += temp[i];
            }

            // Scale the contributing weights by the number of moves factored into. 
            // Note taking the sqrt makes the contributing weights comparable to the score weights.
            double contWeightDiv = Math.Pow(movesOfInterest.Count, 1.0 / 2.0);
            for (int j = 0; j <  contWeights.Count; j++)
            {
                contWeights[j] /= contWeightDiv;
            }

            List<double> highlightFactors = GetHighlightFactors(contWeights, movesOfInterest, 8, 8,/*6, 6,*/ inverse);

            // Stop highlight factors at the latest index we scored
            int maxIndex = movesOfInterest.Max();
            highlightFactors = highlightFactors.GetRange(0, maxIndex + 1);

            float baseWidth = TraceLineWidth * 1.5f;

            DrawColor highlight;
            DrawColor baseColor;
            GetHabitHighlitingColors(out highlight, out baseColor);
            DrawColor highlight2 = highlight;
            if (inverse) 
            { 
                // Show the scored draws in the inverted highlight color
                highlight2 = ThemeColors.InvertColor(highlight2); 
            }

            int startScoredMoves = movesOfInterest.Min();
            for (int j = 0; j < startScoredMoves; j++)
            {
                mScoredDraws[j].DrawWithHighlight(highlightFactors[j], highlight, baseColor, baseWidth);
            }
            for (int j = startScoredMoves; j < highlightFactors.Count; j++)
            {
                mScoredDraws[j].DrawWithHighlight(highlightFactors[j], highlight2, baseColor, baseWidth);
            }

            List<double> hfTemp = new List<double>(highlightFactors);
            for (int j = 0; j < hfTemp.Count; j++)
            {
                int minIndex = hfTemp.IndexOf(hfTemp.Min());
                mScoredDraws[minIndex].ReAddToLayer();

                // Make the minIndex a large value, so it's no longer the min. The normal max would be 1.
                hfTemp[minIndex] = 999;
            }

            ShowDrawHistory(true, 0, highlightFactors.Count, highlightFactors, highlight, baseColor, highlight2, startScoredMoves);

            return true;
        }

        void GetMostExtremeScoringIndexes(int count, out int startIndex)
        {
            startIndex = 0;
            int startIndexMin = 0;
            int startIndexMax = 0;

            int searchStartIndex = 0;
            if (mPlayMode == PlayMode.Continuity)
            {
                searchStartIndex = mContinuityScoredDrawFirstMoveThisGame;
            }

            if (mScoredDraws.Count - searchStartIndex <= count)
            {
                return;
            }

            List<double> startIndexScores = new List<double>();

            // Get the scores for each sequence of count
            for (int j = searchStartIndex; j < mScoredDraws.Count - count; j++)
            {
                double total = 0;
                for (int i = j; i < count + j; i++)
                {
                    total += mScoredDraws[i].score;
                }
                startIndexScores.Add(total);
            }

            // Get the max or min score sequence
            double max = 0;
            double min = 0;
            for (int j = 0; j < startIndexScores.Count; j++)
            {
                if (startIndexScores[j] >= max)
                {
                    startIndexMax = j + searchStartIndex;
                    max = startIndexScores[j];
                }
                if (startIndexScores[j] <= min)
                {
                    startIndexMin = j + searchStartIndex;
                    min = startIndexScores[j];
                }
            }

            if (Math.Abs(min) > Math.Abs(max))
            {
                startIndex = startIndexMin;
            }
            else
            {
                startIndex = startIndexMax;
            }
        }

        bool IsInContinuitySeries()
        {
            return mPlayMode == PlayMode.Continuity && mIsContinuityStarted;
        }
    }
}