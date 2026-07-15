using CocosSharp;
using Java.Util.Functions;
using MindMove.Cocos;
using Plugin.StoreReview;
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Threading;

namespace MindMove.Cocos
{
    class TitleScene : CCScene
    {
        public static string SingleGameMode = "Mind Stream";

        CCLayer mLayer;

        readonly float mSF;

        const float TitleSizeBase = 0.0665f;
        float TitleSize;
        const float InstructionSizeBase = 0.04f;
        float InstructionSize;
        const float TutorialInstructionSizeBase = .038f;
        float TutorialInstructionSize;

        DrawNode mMenuOverlay;
        DrawNode mMenuLockedModeOverlay;
        DrawLabel mModesMenu;
        List<string> cModes = null;

        int mNumLockedModes = 0;

        const float DemoTop = 0.54f;
        const float DemoBottom = 0.46f;
        const float ColorPickerBottom = 0.3f;
        const float MenuHeightPerItem = (DemoBottom / 3f) * 0.98f;
        float mMenuBottom = MenuHeightPerItem * 3;

        DrawColor mBrightBackground;

        DrawNode mTutorialOverlay;
        DrawLabel mTutorialHeaderText;

        DrawLabel mMessageText;

        DrawNode mColorPickerOverlay;
        List<DrawNode> mColorPickerButtons;
        List<ThemeColors> mColorPickerThemeColors;
        List<DrawPoint> mColorPickerButtonRelativeCoords;
        List<DrawLabel> mColorPickerTextItems;

        const float ColorThemeButtonRadiusBase = 0.0375f;
        float ColorThemeButtonRadius;
        const float ColorThemeButtonRadiusMult = 2.5f;
        const float ColorThemeButtonSingleRandomizerRadiusBaseMult = ColorThemeButtonRadiusMult / 2f;
        ThemeColors mThemeColors;
        int mThemeColorIndex = 0;
        const string ThemeColorFile = "SavedThemeColor.xml";
        DrawNode mColorThemeButton;
        DrawLabel mDetailsButton;

        DrawLabel mTitleText;
        DrawLabel mPlayText;
        float mTitleCircleRadius;
        DrawPoint mTitleCircleCenter;
        float mPulseButtonPlayRadius;
        DrawPoint mPulseButtonPlayCenter;
        float mPulseButtonPlayHitRadius;

        float mIdleGlitterRadius;

        RankBadge mTitleBackground;
        RankBadge mPlayBackground;

        DrawNode mPlayBackgroundCircle;
        DrawNode mTitleBackgroundCircle;

        long mTicksLastPulsate = 0;
        Pulsate mPulsate;

        Random mRnd;
        DrawRect mDemoBounds;

        DrawLabel mSoundLabel;
        int mSoundVersion = 0;

        bool mUpdateThreadLocked = true;

        public TitleScene(CCGameView gameView) : base(gameView)
        {
            mSF = MainActivity.Current.ScalingFactor();
            mThemeColorIndex = MainActivity.GetSavedIntWithDefaultValue(0, ThemeColorFile);
            mSoundVersion = MainActivity.SoundEffects.GetSoundVersion();
            mRnd = new Random();
            mPulsate = new Pulsate();
            mIdleGlitterDots = new List<GlitterDot>();
            Init();
            MainActivity.GameWasLoaded = true;
        }

        public void Init()
        {
            mUpdateThreadLocked = true;
            mLayer = new CCLayer();
            this.AddLayer(mLayer);

            TitleSize = mLayer.ContentSize.Width * TitleSizeBase;
            InstructionSize = mLayer.ContentSize.Width * InstructionSizeBase;
            TutorialInstructionSize = mLayer.ContentSize.Width * TutorialInstructionSizeBase;
            ColorThemeButtonRadius = mLayer.ContentSize.Width * ColorThemeButtonRadiusBase;

            if (mLayer.ContentSize.Height / mLayer.ContentSize.Width < SharedParams.SizeAdjustRatioThreshold)
            {
                float mult = (mLayer.ContentSize.Height / mLayer.ContentSize.Width) / SharedParams.SizeAdjustRatioThreshold;
                TitleSize *= mult;
                InstructionSize *= mult;
                TutorialInstructionSize *= mult;
                ColorThemeButtonRadius *= mult;
            }

            mIdleGlitterRadius = (float)Math.Sqrt((double)mLayer.ContentSize.Width * mLayer.ContentSize.Height) * 0.0044f;

            mTitleCircleCenter = new DrawPoint(mLayer.ContentSize.Width / 2.0f, mLayer.ContentSize.Height * (DemoTop + (DemoBottom * 0.54f)));
            mTitleCircleRadius = mLayer.ContentSize.Width * 0.25f;

            mPulseButtonPlayCenter = new DrawPoint(mLayer.ContentSize.Width / 2.0f, mLayer.ContentSize.Height * (DemoBottom - (DemoBottom * 0.5f)));
            mPulseButtonPlayRadius = mTitleCircleRadius * 0.8f;
            mPulseButtonPlayHitRadius = mPulseButtonPlayRadius * 1.25f;

            mMenuOverlay = null;
            mMenuLockedModeOverlay = null;
            mTutorialOverlay = null;
            mColorPickerOverlay = null;
            mModesMenu = null;
            mTutorialHeaderText = null;
            mSoundLabel = null;

            mThemeColors = new ThemeColors(mThemeColorIndex);
            mThemeColors.CreateBackground(ref mLayer);

            DrawColor tempBright = ThemeColors.AdjustColorsForMultipler(1.8, mThemeColors.BackgroundColor);
            mBrightBackground = new DrawColor(tempBright.R, tempBright.G, tempBright.B, 209);

            mDemoBounds = new DrawRect(mLayer.ContentSize.Width * 0.2222f, mLayer.ContentSize.Height * DemoBottom, mLayer.ContentSize.Width * 0.5555f, mLayer.ContentSize.Height * (DemoTop - DemoBottom));
            
            CreateTitle();
            CreateTouchListener();

            mIdleGlitterDots.Clear();

            SetPlayMenu();

            mUpdateThreadLocked = false;
        }

        void SetPlayMenu()
        {
            mNumLockedModes = 0;

            if (cModes != null) { cModes.Clear(); cModes = null; }
            cModes = new List<string>();
            cModes.Add(SingleGameMode);

            string voidStorm = SharedParams.VoidStorm;
            if (!SharedParams.IsUnlockedVoidStorm)
            {
                voidStorm += ": score " + SharedParams.VoidStormUnlock;
                mNumLockedModes++;
            }
            cModes.Add(voidStorm);

            string streamOfCont = SharedParams.StreamOfContinuity;
            if (!SharedParams.IsUnlockedContinuity)
            {
                streamOfCont += ": score " + SharedParams.ContinuityUnlock;
                mNumLockedModes++;
            }
            cModes.Add(streamOfCont);

            mMenuBottom = cModes.Count * MenuHeightPerItem;
        }

        public override void OnEnter()
        {
            mUpdateThreadLocked = true;
            base.OnEnter();
            CreateTitle();
            SetPlayMenu();
            ClearIdleGlitter();
            mUpdateThreadLocked = false;
        }

        public override void OnExit()
        {
            mUpdateThreadLocked = true;
            base.OnExit();
            TerminateBadges();
            RemoveModeMenuOverlay();
            ClearIdleGlitter();
            mUpdateThreadLocked = false;
        }

        void ClearIdleGlitter()
        {
            try
            {
                foreach (GlitterDot dot in mIdleGlitterDots)
                {
                    dot.Clear();
                }
                mIdleGlitterDots.Clear();
            }
            catch { }
        }

        private void CreateTitle()
        {
            TerminateBadges();

            float badgeRadius = mLayer.ContentSize.Height / 4.625f;
            mTitleBackground = new RankBadge(ref mLayer, badgeRadius, mTitleCircleCenter, mThemeColors, 5, false, draw:true, dotsOnly: true);

            if (mTitleBackgroundCircle != null)
            {
                mLayer.RemoveChild(mTitleBackgroundCircle);
                mTitleBackgroundCircle = null;
            }
            mTitleBackgroundCircle = new DrawNode();
            mLayer.AddChild(mTitleBackgroundCircle);
            mTitleBackgroundCircle.DrawSolidCircle(mTitleCircleCenter, badgeRadius / 2.222f, mBrightBackground);

            mTitleText = new DrawLabel("", "Arial", TitleSize*1.45f);
            mTitleText.Color = mThemeColors.Text;
            mTitleText.PositionX = mTitleCircleCenter.X;
            mTitleText.PositionY = mTitleCircleCenter.Y;
            mTitleText.HorizontalAlignment = CCTextAlignment.Center;
            mLayer.AddChild(mTitleText);
            ShowTitleText();

            // Demo bounds
            DrawColor boxLineColor = ThemeColors.AverageColors(mThemeColors.DistinctColorA, mThemeColors.DistinctColorB);
            DrawNode demoBox = new DrawNode();
            mLayer.AddChild(demoBox);
            demoBox.DrawLine(
                from: new DrawPoint(mLayer.ContentSize.Width * 0.05f, mDemoBounds.MaxY + 1*mSF),
                to: new DrawPoint(mLayer.ContentSize.Width * 0.95f, mDemoBounds.MaxY + 1*mSF),
                color: boxLineColor,
                lineWidth: 1*mSF,
                lineCap: CCLineCap.Round);
            demoBox.DrawLine(
                from: new DrawPoint(mLayer.ContentSize.Width * 0.05f, mDemoBounds.MinY - 1*mSF),
                to: new DrawPoint(mLayer.ContentSize.Width * 0.95f, mDemoBounds.MinY - 1*mSF),
                color: boxLineColor,
                lineWidth: 1 * mSF,
                lineCap: CCLineCap.Round);
            demoBox.DrawLine(
                from: new DrawPoint(mDemoBounds.MinX, mDemoBounds.MinY + (mDemoBounds.Size.Height * 0.125f)),
                to: new DrawPoint(mDemoBounds.MinX, mDemoBounds.MaxY - (mDemoBounds.Size.Height * 0.125f)),
                color: mThemeColors.MiddleColor,
                lineWidth: 1 * mSF,
                lineCap: CCLineCap.Round);
            demoBox.DrawLine(
                from: new DrawPoint(mDemoBounds.MaxX, mDemoBounds.MinY + (mDemoBounds.Size.Height * 0.125f)),
                to: new DrawPoint(mDemoBounds.MaxX, mDemoBounds.MaxY - (mDemoBounds.Size.Height * 0.125f)),
                color: mThemeColors.MiddleColor,
                lineWidth: 1 * mSF,
                lineCap: CCLineCap.Round);

            // Play button
            badgeRadius *= (6f / 7f);
            mPlayBackground = new RankBadge(ref mLayer, badgeRadius, mPulseButtonPlayCenter, mThemeColors, 5, false, draw:true);

            if (mPlayBackgroundCircle != null)
            {
                mLayer.RemoveChild(mPlayBackgroundCircle);
                mPlayBackgroundCircle = null;
            }
            mPlayBackgroundCircle = new DrawNode();
            mLayer.AddChild(mPlayBackgroundCircle);
            mPlayBackgroundCircle.DrawSolidCircle(mPulseButtonPlayCenter, badgeRadius / 2.222f, mBrightBackground);

            mPlayText = new DrawLabel("Play", "Arial", TitleSize*1.8f);
            mPlayText.PositionX = mPulseButtonPlayCenter.X;
            mPlayText.PositionY = mPulseButtonPlayCenter.Y;
            mPlayText.HorizontalAlignment = CCTextAlignment.Center;
            mPlayText.Color = mThemeColors.Text;
            mLayer.AddChild(mPlayText);

            // Slogan message
            string message = "Will your mindful changes\nperform better\nthan mindlessness?";

            if (mMessageText != null) { mLayer.RemoveChild(mMessageText); mMessageText = null; }
            mMessageText = new DrawLabel(message, "Arial", InstructionSize*0.98f);
            mMessageText.Color = mThemeColors.Text;
            mMessageText.PositionX = mLayer.ContentSize.Width / 2.0f;
            mMessageText.PositionY = mDemoBounds.MidY;
            mMessageText.HorizontalAlignment = CCTextAlignment.Center;
            mLayer.AddChild(mMessageText);

            // Details button
            float size = TitleSize * 1.8f;
            mDetailsButton = new DrawLabel("?", "Arial", size);
            mDetailsButton.PositionX = mDemoBounds.MaxX + ((mLayer.ContentSize.Width - mDemoBounds.MaxX) / 2);
            mDetailsButton.PositionY = mLayer.ContentSize.Height * 0.5f;
            mDetailsButton.HorizontalAlignment = CCTextAlignment.Center;
            mDetailsButton.Color = mThemeColors.TextColor1;

            float rad = size * 0.4545f;
            DrawNode detailsUnderlay = new DrawNode();
            detailsUnderlay.DrawSolidCircle(mDetailsButton.Position, rad, ThemeColors.ReduceColorForMultiplierCheap(0.4, mThemeColors.AveragedColor));
            detailsUnderlay.DrawCircle(mDetailsButton.Position, rad, mThemeColors.Color1);
            detailsUnderlay.DrawCircle(mDetailsButton.Position, rad+1, mThemeColors.Color1);
            detailsUnderlay.DrawCircle(mDetailsButton.Position, rad-1, mThemeColors.Color1);
            mLayer.AddChild(detailsUnderlay);
            mLayer.AddChild(mDetailsButton);

            SetSoundLabel();
            DrawColorThemeChangeButton();

            this.Schedule();
        }

        void SetSoundLabel()
        {
            if (mSoundLabel != null)
            {
                mLayer.RemoveChild(mSoundLabel);
                mSoundLabel = null;
            }

            string text = "Sound";
            float widthOffset = 0.9f;
            if (mSoundVersion == Sound.SoundEffectsOnly)
            {
            }
            else if (mSoundVersion == Sound.SoundOff)
            {
                text += " off";
            }

            mSoundLabel = new DrawLabel(text, "Arial", InstructionSize*0.9f);
            mSoundLabel.PositionX = mLayer.ContentSize.Width * widthOffset;
            mSoundLabel.PositionY = mLayer.ContentSize.Height * 0.025f;
            mSoundLabel.HorizontalAlignment = CCTextAlignment.Left;
            
            DrawColor color = mThemeColors.TextColor1;
            if (mSoundVersion == 0)
            {
                color = ThemeColors.AdjustColorsForMultipler(0.75, color.R, color.G, color.B);
            }
            else
            {
                color = ThemeColors.AdjustColorsForMultipler(0.9, color.R, color.G, color.B);
            }
            mSoundLabel.Color = color;

            mLayer.AddChild(mSoundLabel);
        }

        private void DrawColorThemeChangeButton(bool pulsateMode = false)
        {
            DrawColorPickerThemeButton(
                ref mColorThemeButton, 
                mThemeColors,
                (mDemoBounds.MinX / 2) / mLayer.ContentSize.Width,
                0.5f,
                false,
                pulsateMode: pulsateMode);
        }

        private void CreateTouchListener()
        {
            var touchListener = new CCEventListenerTouchAllAtOnce();
            touchListener.OnTouchesBegan = HandleTouchesBegan;
            mLayer.AddEventListener(touchListener);
        }

        private void HandleTouchesBegan(List<CCTouch> arg1, CCEvent arg2)
        {
            DrawPoint location = arg1[0].Location;

            if (mTutorialOverlay != null)
            {
                RemoveTutorialOverlay();
                return;
            }

            if (mColorPickerOverlay != null && mColorPickerButtonRelativeCoords != null)
            {
                ThemeColors.ClearSingleRandomColorSettings();
                int indexFinalPickerButton = ThemeColors.GetNumColorThemes();

                int unlockedColorCount = SharedParams.GetNumUnlockedColorThemes();

                float radius = ColorThemeButtonRadius * ColorThemeButtonRadiusMult;
                for (int j = 0; j < mColorPickerButtonRelativeCoords.Count; j++)
                {
                    if (j <= indexFinalPickerButton)
                    {
                        if (location.X > mColorPickerButtonRelativeCoords[j].X - radius && location.X < mColorPickerButtonRelativeCoords[j].X + radius &&
                            location.Y > mColorPickerButtonRelativeCoords[j].Y - radius && location.Y < mColorPickerButtonRelativeCoords[j].Y + radius)
                        {
                            if (j < unlockedColorCount)
                            {
                                mThemeColorIndex = mColorPickerThemeColors[j].GetColorIndex();
                                MainActivity.SaveIntValue(mThemeColorIndex, ThemeColorFile);
                                RefreshColorPickerOverlay();
                            }
                            return;
                        }
                    }
                    else
                    {
                        radius = ColorThemeButtonRadius * ColorThemeButtonSingleRandomizerRadiusBaseMult;
                        if (location.X > mColorPickerButtonRelativeCoords[j].X - radius && location.X < mColorPickerButtonRelativeCoords[j].X + radius &&
                            location.Y > mColorPickerButtonRelativeCoords[j].Y - radius && location.Y < mColorPickerButtonRelativeCoords[j].Y + radius)
                        {
                            int saveButtonStart = indexFinalPickerButton + 4;
                            if (j >= saveButtonStart)
                            {
                                ThemeColors.SaveColorTheme(mThemeColors, j - saveButtonStart);
                                mThemeColorIndex = mThemeColors.GetColorIndex() + 1 + (j - saveButtonStart);
                                MainActivity.SaveIntValue(mThemeColorIndex, ThemeColorFile);
                                RefreshColorPickerOverlay();
                                return;
                            }
                            else
                            {
                                ThemeColors.SetSingleRandomColorToChange(j - (indexFinalPickerButton + 1));
                                RefreshColorPickerOverlay();
                                return;
                            }
                        }
                    }
                }

                RemoveColorPickerOverlay();
                return;
            }

            if ((location.X > (mDemoBounds.MinX / 2) - TitleSize*1.33f) &&
                (location.X < (mDemoBounds.MinX / 2) + TitleSize*1.33f) &&
                (location.Y > (mDemoBounds.MidY) - TitleSize) &&
                (location.Y < (mDemoBounds.MidY) + TitleSize))
            {
                if (mMenuOverlay != null)
                {
                    RemoveModeMenuOverlay();
                }
                else
                {
                    ShowColorPickerOverlay();
                }
            }
            else if (mMenuOverlay == null &&
                (location.X > mDetailsButton.PositionX - TitleSize) &&
                (location.X < mDetailsButton.PositionX + TitleSize) &&
                (location.Y > mDetailsButton.PositionY - TitleSize) &&
                (location.Y < mDetailsButton.PositionY + TitleSize))
            {
                MainActivity.Current.ShowHelp();
            }
            else if (mTitleBackgroundCircle.BoundingRect.ContainsPoint(location))
            {
                ShowPoemOverlay();
            }
            else if (location.Y < mLayer.ContentSize.Height * mMenuBottom &&
                location.Y > mLayer.ContentSize.Height * 0.01f) 
            {
                if (mMenuOverlay == null)
                {
                    if (Math.Pow(location.X - mPulseButtonPlayCenter.X, 2) + Math.Pow(location.Y - mPulseButtonPlayCenter.Y, 2) < Math.Pow(mPulseButtonPlayHitRadius, 2))
                    {
                        ShowModeMenuOverlay(cModes);
                    }
                    else if (location.X > mLayer.ContentSize.Width * 0.82f && location.Y < mLayer.ContentSize.Height * 0.06f)
                    {
                        mSoundVersion++;
                        if (mSoundVersion > Sound.MaxSoundVersion)
                        {
                            mSoundVersion = Sound.SoundOff;
                        }

                        SetSoundLabel();

                        MainActivity.SoundEffects.SetSoundVersion(mSoundVersion);
                        MainActivity.SoundEffects.MenuItem();
                    }
                }
                else
                {
                    for (int j = 1; j <= cModes.Count; j++)
                    {
                        float checkLine = (mLayer.ContentSize.Height * mMenuBottom) - ((mLayer.ContentSize.Height * mMenuBottom) * ((float)j / (float)cModes.Count));
                        if (location.Y > checkLine)
                        {
                            if ((cModes[j - 1].StartsWith(SharedParams.VoidStorm) && !SharedParams.IsUnlockedVoidStorm) ||
                                (cModes[j - 1].StartsWith(SharedParams.StreamOfContinuity) && !SharedParams.IsUnlockedContinuity))
                            {
                                MainActivity.SoundEffects.HitDormantMine();
                                return;
                            }
                            else
                            {
                                MainActivity.SoundEffects.MenuItem();
                            }

                            StartGame(j - 1);
                            break;
                        }
                    }
                }
            }
            else
            {
                RemoveModeMenuOverlay();
            }
        }

        void StartGame(int modeIndex)
        {
            string mode = cModes[modeIndex];
            StartGame(mode);
        }

        void StartGame(string mode)
        {
            var newScene = new GameScene(GameController.GameView, mode, mThemeColors);
            GameController.PushScene(newScene);
        }

        void TerminateBadges()
        {
            if (mTitleBackground != null)
            {
                mTitleBackground.Deactivate();
                mTitleBackground = null;
            }
            if (mPlayBackground != null)
            {
                mPlayBackground.Deactivate();
                mPlayBackground = null;
            }
        }

        string GetPoemText(out float fontSize)
        {
            string text = "~ Scribbling Game ~\n";

            fontSize = InstructionSize;

            text += "\nCandescent ribbons in ruby to teal hues";
            text += "\nstream and curl from here to the canvas -";

            text += "\n\na sprawl of loops, jitters, and lines";
            text += "\nstriving for the shape of randomness.";

            text += "\n\nThe hand flits in a dance of inner jolts,";
            text += "\na dragonfly-like maze of woven strokes;";

            text += "\n\nstill, this torrent moves along grooves";
            text += "\nand doodling funnels into form—";

            text += "\n\nthe bulbed curves, curly-ques in threes;";
            text += "\ncadence of long stripes, little squiggles;";

            text += "\n\nthe corners are barely trodden,";
            text += "\nan odd bias against diagonals—";

            text += "\n\nis scribbling then a slippery game,";
            text += "\nas signature shape continually flows?";

            return text;
        }

        void ShowPoemOverlay()
        {
            float centerY = mTitleBackgroundCircle.BoundingRect.Center.Y - mLayer.ContentSize.Height * 0.067f;

            mTutorialOverlay = new DrawNode();
            mLayer.AddChild(mTutorialOverlay);
            mTutorialOverlay.DrawLine(
                from: new DrawPoint(0.02f * mLayer.ContentSize.Width, centerY),
                to: new DrawPoint(mLayer.ContentSize.Width * 0.98f, centerY),
                color: mBrightBackground,
                lineWidth: (mLayer.ContentSize.Height * DemoTop * 0.5f));

            float fontSize;
            string headerText = GetPoemText(out fontSize);
            mTutorialHeaderText = new DrawLabel(headerText, "Arial", fontSize * 1f);
            mTutorialHeaderText.Color = ThemeColors.AdjustColorsForMultipler(1.1, mThemeColors.Text);
            mTutorialHeaderText.PositionX = mLayer.ContentSize.Width * 0.5f;
            mTutorialHeaderText.PositionY = centerY;
            mTutorialHeaderText.HorizontalAlignment = CCTextAlignment.Center;
            mLayer.AddChild(mTutorialHeaderText);

            ShowTitleText(false);
        }

        void ShowTitleText(bool show = true)
        {
            if (show) 
            { 
                mTitleText.Text = "Mind\nUnbind";
                mTitleBackgroundCircle.Visible = true;
            }
            else 
            { 
                mTitleText.Text = "";
                mTitleBackgroundCircle.Visible = false;
            }
        }

        void RemoveTutorialOverlay()
        {
            if (mTutorialOverlay != null)
            {
                mLayer.RemoveChild(mTutorialOverlay);
                mTutorialOverlay = null;
            }
            if (mTutorialHeaderText != null)
            {
                mLayer.RemoveChild(mTutorialHeaderText);
                mTutorialHeaderText = null;
            }

            ShowTitleText();
        }

        DrawNode mUnlockColorThemesButton = null;

        void ShowColorPickerOverlay()
        {
            mColorPickerButtons = new List<DrawNode>();
            mColorPickerThemeColors = new List<ThemeColors>();
            mColorPickerButtonRelativeCoords = new List<DrawPoint>();
            mColorPickerTextItems = new List<DrawLabel>();

            DrawPoint start = new DrawPoint(0, mLayer.ContentSize.Height);
            DrawPoint end = new DrawPoint(mLayer.ContentSize.Width, mLayer.ContentSize.Height);

            mColorPickerOverlay = new DrawNode();
            mLayer.AddChild(mColorPickerOverlay);
            mColorPickerOverlay.DrawLine(
                from: start,
                to: end,
                color: mThemeColors.OverlayDark,
                lineWidth: (mLayer.ContentSize.Height - (ColorPickerBottom * mLayer.ContentSize.Height)));

            int numThemes = ThemeColors.GetNumColorThemes() + 1;
            int randomButtonIndex = 0;

            int numColorRows = (int)Math.Ceiling((double)numThemes / 3.0);
            int numRowsUnlocked = SharedParams.GetNumUnlockedColorThemes() / 3;

            int numPerRow = 3;
            const float yBase = 0.9383f;
            for (int j = 0; j < numThemes; j++)
            {
                float x;
                int jMod = j % numPerRow;
                if (jMod == 0) { x = 0.2f; }
                else if (jMod == 1) { x = 0.5f; }
                else { x = 0.8f; }

                float y = yBase - ((j / numPerRow) * 2 * (1f - yBase));

                ThemeColors tc = new ThemeColors(j, true);
                DrawNode button = null;

                bool useGray = false;
                if (tc.IsRandomColorTheme()) 
                { 
                    useGray = true;
                    randomButtonIndex = j;
                }

                DrawColorPickerThemeButton(ref button, tc, x, y, true, useGray: useGray);

                mColorPickerButtonRelativeCoords.Add(new DrawPoint(x * mLayer.ContentSize.Width, y * mLayer.ContentSize.Height));
                mColorPickerThemeColors.Add(tc);
                mColorPickerButtons.Add(button);

                if (numRowsUnlocked < numColorRows && j == (numRowsUnlocked * numPerRow) - 1)
                {
                    float y0 = (y - (1f - yBase)) * mLayer.ContentSize.Height;
                    button.DrawLine(
                        from: new DrawPoint(mLayer.ContentSize.Width * 0.1f, y0),
                        to: new DrawPoint(mLayer.ContentSize.Width * 0.9f, y0),
                        color: mThemeColors.AveragedColor,
                        lineWidth: 2 * mSF);
                }
            }

            if (mThemeColors.IsRandomColorTheme())
            {
                float radiusSmallPickerButton = ColorThemeButtonRadius * ColorThemeButtonSingleRandomizerRadiusBaseMult;
                float radiusPickerButton = ColorThemeButtonRadius * ColorThemeButtonRadiusMult;

                float x = mColorPickerButtonRelativeCoords[randomButtonIndex].X;
                float y = mColorPickerButtonRelativeCoords[randomButtonIndex].Y - (radiusPickerButton + radiusSmallPickerButton*1.5f);

                DrawNode button1 = new DrawNode();
                DrawPoint pt1 = new DrawPoint(x - radiusPickerButton*1.2f, y);
                button1.DrawSolidCircle(pt1, radiusSmallPickerButton, mThemeColors.Color1);
                mColorPickerButtonRelativeCoords.Add(pt1);
                mColorPickerButtons.Add(button1);
                mLayer.AddChild(button1);

                DrawNode button2 = new DrawNode();
                DrawPoint pt2 = new DrawPoint(x, y);
                button2.DrawSolidCircle(pt2, radiusSmallPickerButton, mThemeColors.Color2);
                mColorPickerButtonRelativeCoords.Add(pt2);
                mColorPickerButtons.Add(button2);
                mLayer.AddChild(button2);

                DrawNode button3 = new DrawNode();
                DrawPoint pt3 = new DrawPoint(x + radiusPickerButton*1.2f, y);
                button3.DrawSolidCircle(pt3, radiusSmallPickerButton, mThemeColors.Color3);
                mColorPickerButtonRelativeCoords.Add(pt3);
                mColorPickerButtons.Add(button3);
                mLayer.AddChild(button3);

                DrawPoint ptSave = new DrawPoint(mLayer.ContentSize.Width * 0.5f, pt3.Y);
                DrawLabel saveColorText = new DrawLabel("Save", "Arial", InstructionSize);
                saveColorText.Color = mThemeColors.TextColor1;
                saveColorText.PositionX = ptSave.X;
                saveColorText.PositionY = ptSave.Y;
                saveColorText.HorizontalAlignment = CCTextAlignment.Center;
                mLayer.AddChild(saveColorText);
                mColorPickerButtonRelativeCoords.Add(ptSave);
                mColorPickerTextItems.Add(saveColorText);

                if (ThemeColors.GetNumCustomThemes() >= 1)
                {
                    DrawPoint ptSave2 = new DrawPoint(mLayer.ContentSize.Width * 0.8f, pt3.Y);
                    DrawLabel save2ColorText = new DrawLabel("Save", "Arial", InstructionSize);
                    save2ColorText.Color = mThemeColors.TextColor1;
                    save2ColorText.PositionX = ptSave2.X;
                    save2ColorText.PositionY = ptSave2.Y;
                    save2ColorText.HorizontalAlignment = CCTextAlignment.Center;
                    mLayer.AddChild(save2ColorText);
                    mColorPickerButtonRelativeCoords.Add(ptSave2);
                    mColorPickerTextItems.Add(save2ColorText);
                }
            }
            
            string text = "";
            float yOffset = (mLayer.ContentSize.Height * 0.033f);
            float instructionFontSize = InstructionSize;
            if (numRowsUnlocked < numColorRows)
            {
                text += "Unlock another row of themes";
                text += "\nfor every " + SharedParams.NumTitleUnlocksPerAward + " titles achieved.\n\n";
                text += "Titles achieved: " + SharedParams.TotalTitleUnlocks;
                yOffset += InstructionSize*4;
                instructionFontSize *= 1.3f;

                DrawPoint ptUnlock = new DrawPoint(mLayer.ContentSize.Width * 0.5f, yOffset + (ColorPickerBottom * mLayer.ContentSize.Height));

                mUnlockColorThemesButton = new DrawNode();
                mLayer.AddChild(mUnlockColorThemesButton);
                mUnlockColorThemesButton.DrawLine(
                    from: new DrawPoint(0, ptUnlock.Y),
                    to: new DrawPoint(mLayer.ContentSize.Width, ptUnlock.Y),
                    color: new DrawColor(0, 0, 0, 128),
                    lineWidth: instructionFontSize * 4);
                mColorPickerButtonRelativeCoords.Add(ptUnlock);
            }

            DrawLabel colorPickerText = new DrawLabel(text, "Arial", instructionFontSize);
            colorPickerText.Color = ThemeColors.AdjustColorsForMultipler(1.1, mThemeColors.TextColor1);
            colorPickerText.PositionX = mLayer.ContentSize.Width * 0.5f;
            colorPickerText.PositionY = yOffset + (ColorPickerBottom * mLayer.ContentSize.Height);
            colorPickerText.HorizontalAlignment = CCTextAlignment.Center;
            mLayer.AddChild(colorPickerText);
            mColorPickerTextItems.Add(colorPickerText);
        }

        private void RefreshColorPickerOverlay()
        {
            RemoveColorPickerOverlay();
            this.RemoveAllChildren();
            Init();
            ShowColorPickerOverlay();
        }

        void RemoveColorPickerOverlay()
        {
            if (mColorPickerOverlay != null)
            {
                mLayer.RemoveChild(mColorPickerOverlay);
                mColorPickerOverlay = null;
            }

            if (mColorPickerButtons != null)
            {
                for (int j = 0; j < mColorPickerButtons.Count; j++)
                {
                    mLayer.RemoveChild(mColorPickerButtons[j]);
                    mColorPickerButtons[j] = null;
                }
                mColorPickerButtons.Clear();
                mColorPickerButtons = null;
            }

            if (mColorPickerTextItems != null)
            {
                for (int j = 0; j < mColorPickerTextItems.Count; j++)
                {
                    mLayer.RemoveChild(mColorPickerTextItems[j]);
                    mColorPickerTextItems[j] = null;
                }
                mColorPickerTextItems.Clear();
                mColorPickerTextItems = null;
            }

            if (mColorPickerButtonRelativeCoords != null)
            {
                mColorPickerButtonRelativeCoords.Clear();
                mColorPickerButtonRelativeCoords = null;
            }

            if (mColorPickerThemeColors != null)
            {
                for (int j = 0; j < mColorPickerThemeColors.Count; j++)
                {
                    mColorPickerThemeColors[j] = null;
                }
                mColorPickerThemeColors.Clear();
                mColorPickerThemeColors = null;
            }
            
            if (mUnlockColorThemesButton != null)
            {
                mLayer.RemoveChild(mUnlockColorThemesButton);
                mUnlockColorThemesButton = null; 
            }
        }

        void DrawColorPickerThemeButton(ref DrawNode button, ThemeColors colors, float x, float y, bool drawBackground, bool pulsateMode = false, bool useGray = false)
        {
            if (button != null)
            {
                button.Clear();
                button.Cleanup();
                button.Visible = false;
            }
            else
            {
                button = new DrawNode();
            }

            if (button.GetInternalNode().Parent != mLayer)
            {
                mLayer.AddChild(button);
            }

            DrawColor color1 = colors.Color1;
            DrawColor color2 = colors.Color2;
            DrawColor color3 = colors.Color3;
            DrawColor color12 = colors.AveragedColor12;
            DrawColor color13 = colors.AveragedColor13;
            DrawColor color23 = colors.AveragedColor23;

            x *= mLayer.ContentSize.Width;
            y *= mLayer.ContentSize.Height;

            if (useGray)
            {
                color1 = new DrawColor(128, 128, 128);
                color2 = new DrawColor(128, 128, 128);
                color3 = new DrawColor(128, 128, 128);
                color12 = new DrawColor(128, 128, 128);
                color13 = new DrawColor(128, 128, 128);
                color23 = new DrawColor(128, 128, 128);
            }

            if (pulsateMode)
            {
                float mult = 1.15f;
                DrawColor temp1 = color1;
                DrawColor temp2 = color2;
                DrawColor temp3 = color3;
                color1 = ThemeColors.AdjustColorsForMultipler(mPulsate.GetReducedScaleFactor(mult), temp1.R, temp1.G, temp1.B);
                color2 = ThemeColors.AdjustColorsForMultipler(mPulsate.GetReducedScaleFactor(mult), temp2.R, temp2.G, temp2.B);
                color3 = ThemeColors.AdjustColorsForMultipler(mPulsate.GetReducedScaleFactor(mult), temp3.R, temp3.G, temp3.B);
            }

            float radius = ColorThemeButtonRadius;

            if (drawBackground)
            {
                DrawColor background = new DrawColor(colors.BackgroundColor.R, colors.BackgroundColor.G, colors.BackgroundColor.B);
                float bkgRadius = radius * ColorThemeButtonRadiusMult;
                DrawColor averagedColor = colors.AveragedColor;

                if (useGray)
                {
                    background = new DrawColor(128, 128, 128);
                    background = ThemeColors.AdjustColorsForMultipler(0.2, background.R, background.G, background.B);
                    averagedColor = new DrawColor(128, 128, 128);

                    DrawLabel randomColorText = new DrawLabel("?", "Arial", TitleSize*2f);
                    randomColorText.Color = colors.TextColor2;
                    randomColorText.PositionX = x;
                    randomColorText.PositionY = y;
                    randomColorText.HorizontalAlignment = CCTextAlignment.Center;
                    mLayer.AddChild(randomColorText);
                    if (mColorPickerTextItems != null) { mColorPickerTextItems.Add(randomColorText); }
                }

                DrawPoint center = new DrawPoint(x, y);
                button.DrawSolidCircle(center, bkgRadius, background);

                if (mThemeColorIndex == colors.GetColorIndex())
                {
                    button.DrawCircle(center, bkgRadius, averagedColor);
                    button.DrawCircle(center, bkgRadius - (1 * mSF), averagedColor);
                    button.DrawCircle(center, bkgRadius - (2 * mSF), averagedColor);
                }
            }

            float innerRadius = radius - (radius * SharedParams.OneNineth);
            button.DrawSolidCircle(new DrawPoint(x - ColorThemeButtonRadius, y), radius, color12);
            button.DrawSolidCircle(new DrawPoint(x, y), radius, color23);
            button.DrawSolidCircle(new DrawPoint(x + ColorThemeButtonRadius, y), radius, color13);

            button.DrawSolidCircle(new DrawPoint(x - ColorThemeButtonRadius, y), innerRadius, color3);
            button.DrawSolidCircle(new DrawPoint(x, y), innerRadius, color1);
            button.DrawSolidCircle(new DrawPoint(x + ColorThemeButtonRadius, y), innerRadius, color2);

            button.Visible = true;
        }

        void ShowModeMenuOverlay(List<string> cModes)
        {
            string divider = "\n\n\n";
            int lineCount = (3 * cModes.Count) + 4;
            float menuItemSize = (mLayer.ContentSize.Height * mMenuBottom) / (lineCount);
            if (cModes.Count == 1) { menuItemSize *= 0.75f; }

            DrawPoint start = new DrawPoint(0, 0);
            DrawPoint end = new DrawPoint(mLayer.ContentSize.Width, 0);

            mMenuOverlay = new DrawNode();
            mLayer.AddChild(mMenuOverlay);
            mMenuOverlay.DrawLine(
                from: start,
                to: end,
                color: mThemeColors.Overlay,
                lineWidth: mLayer.ContentSize.Height * mMenuBottom);

            mModesMenu = new DrawLabel("", "Arial", menuItemSize);
            mModesMenu.PositionX = mLayer.ContentSize.Width * 0.5f;
            mModesMenu.PositionY = mLayer.ContentSize.Height * (mMenuBottom / 2f);
            mModesMenu.HorizontalAlignment = CCTextAlignment.Center;
            mModesMenu.Color = mThemeColors.TextColor1;

            mModesMenu.Text += divider;
            foreach (string c in cModes)
            {
                mModesMenu.Text += c + divider;
            }

            mLayer.AddChild(mModesMenu);

            if (cModes.Contains(SingleGameMode))
            {
                float height = mMenuBottom * ((float)mNumLockedModes / cModes.Count);
                if (height > 0)
                {
                    mMenuLockedModeOverlay = new DrawNode();
                    mLayer.AddChild(mMenuLockedModeOverlay);
                    mMenuLockedModeOverlay.DrawLine(
                        from: start,
                        to: end,
                        color: new DrawColor(0, 0, 0, 113),
                        lineWidth: mLayer.ContentSize.Height * height);
                }
            }
        }

        void RemoveModeMenuOverlay()
        {
            if (mMenuOverlay != null)
            {
                mLayer.RemoveChild(mMenuOverlay);
                mMenuOverlay = null;
            }
            if (mMenuLockedModeOverlay != null) 
            {
                mLayer.RemoveChild(mMenuLockedModeOverlay);
                mMenuLockedModeOverlay = null;
            }
            if (mModesMenu != null)
            {
                mLayer.RemoveChild(mModesMenu);
                mModesMenu = null;
            }
        }

        readonly List<GlitterDot> mIdleGlitterDots;
        void ProcessIdleGlitter()
        {
            int NumIdleGlitterDots = mTutorialOverlay == null ? 60 : 30;

            long now = DateTime.UtcNow.Ticks;
            if (mIdleGlitterDots.Count == 0) { now -= 540 * TimeSpan.TicksPerMillisecond; }
            while (mIdleGlitterDots.Count < NumIdleGlitterDots)
            {
                DrawPoint pt;
                if (mTutorialOverlay == null && mRnd.NextDouble() < 0.2)
                {
                    pt = new DrawPoint(
                        RandomNumberGenerator.GetInt32(0, (int)mLayer.ContentSize.Width),
                        RandomNumberGenerator.GetInt32((int)mDemoBounds.MinY, (int)mDemoBounds.MaxY));
                }
                else
                {
                    pt = new DrawPoint(
                        RandomNumberGenerator.GetInt32(0, (int)mLayer.ContentSize.Width),
                        RandomNumberGenerator.GetInt32(0, (int)mLayer.ContentSize.Height));

                    if (mTutorialOverlay == null)
                    {
                        if (pt.X < mLayer.ContentSize.Width / 2)
                        {
                            pt.X *= (float)Math.Sqrt(mRnd.NextDouble());
                        }
                        else
                        {
                            pt.X = mLayer.ContentSize.Width - ((mLayer.ContentSize.Width - pt.X) * (float)Math.Sqrt(mRnd.NextDouble()));
                        }
                    }
                }

                long duration = RandomNumberGenerator.GetInt32(2175, 3260) * TimeSpan.TicksPerMillisecond;
                long offsetFromNow = now + RandomNumberGenerator.GetInt32(0, 4350) * TimeSpan.TicksPerMillisecond;
                float radius = mIdleGlitterRadius * (float)(mRnd.NextDouble() + (1 / Math.E));

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
                        mIdleGlitterDots.RemoveAt(j);
                        j--;
                    }
                }
            }
            catch { }
        }

        public override void Update(float dt)
        {
            if (mUpdateThreadLocked) return;

            try
            {
                if (DateTime.UtcNow.Ticks > (mTicksLastPulsate + (SharedParams.PulsatePeriod * TimeSpan.TicksPerMillisecond)))
                {
                    mTicksLastPulsate = DateTime.UtcNow.Ticks;

                    mPulsate.NextScaleFactor();

                    if (mPlayBackground != null) { mPlayBackground.PulsateBadgeRing(mPulsate.GetScaleFactor(), rotateColors:false); }
                    if (mTitleBackground != null) { mTitleBackground.PulsateBadgeRing(mPulsate.GetScaleFactor(), invert:true, rotateColors:false); }

                    DrawColorThemeChangeButton(true);
                }

                ProcessIdleGlitter();
            }
            catch { }
        }
    }
}