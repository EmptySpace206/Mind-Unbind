using Android.App;
using Android.Content.PM;
using Android.OS;
using Android.Runtime;
using Android.Views;
using System.Xml.Serialization;
using System.IO;

using CocosSharp;
using MindMove.Cocos;
using System;

namespace MindMove
{
    [Activity(
        Label = "Mind Unbind",
        MainLauncher = true,
        ScreenOrientation = ScreenOrientation.Portrait,
        Immersive = true
    )]
    public class MainActivity : Activity
    {
        public static MainActivity Current;
        public static Sound SoundEffects;

        // First use file.
        const string PreviouslyLaunched = "PreviouslyLaunched.xml";

        // For debugging (via telemetry) whether the game was loaded correctly.
        // Set to true on complete of loading the title scene.
        public static bool GameWasLoaded = false;

        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            // Init functionality
            Xamarin.Essentials.Platform.Init(this, savedInstanceState);

            Current = this;

            //mIAP_Donation = GetSavedIntWithDefaultValue(0, IAP_ColorThemesSave); 

            ThemeColors.LoadSavedColorThemes();

            SoundEffects = new Sound();

            SharedParams.UpdateUnlocks();

            // Init the game
            SetContentView(Resource.Layout.activity_main);

            SetFullScreenImmersive();

            CCGameView gameView = (CCGameView)FindViewById(Resource.Id.GameView);
            gameView.ViewCreated += LoadGame;

            // Sign in
            int priorLaunch = GetSavedIntWithDefaultValue(0, PreviouslyLaunched);
            if (priorLaunch == 0)
            {
                SaveIntValue(1, PreviouslyLaunched);
            }
        }

        bool ChildActivityActive = false;
        public void ShowHelp()
        {
            ChildActivityActive = true;
            StartActivity(typeof(Help));
        }

        protected override void OnResume()
        {
            base.OnResume();
            SetFullScreenImmersive();
            ChildActivityActive = false;
        }

        protected override void OnPause()
        {
            base.OnPause();
        }

        protected override void OnStop()
        {
            base.OnStop();

            if (!ChildActivityActive)
            {
                this.MoveTaskToBack(true); // Move to the back manually, since CocosSharp freezes on resume. Manual user resume is required.
            }
            if (!GameWasLoaded)
            {
                GameWasLoaded = true; // Only fire this once per session.
            }
        }

        void SetFullScreenImmersive()
        {
            if (this.Window != null)
            {
                this.Window.DecorView.SystemUiVisibility |= (StatusBarVisibility)
                    (SystemUiFlags.HideNavigation |
                    SystemUiFlags.Fullscreen |
                    SystemUiFlags.LayoutHideNavigation |
                    SystemUiFlags.LayoutFullscreen |
                    SystemUiFlags.Immersive |
                    SystemUiFlags.ImmersiveSticky);
            }
        }

        void LoadGame(object sender, EventArgs e)
        {
            CCGameView gameView = sender as CCGameView;

            if (gameView != null)
            {
                gameView.DesignResolution = new CCSizeI(
                    this.Resources.DisplayMetrics.WidthPixels,
                    this.Resources.DisplayMetrics.HeightPixels);
                GameController.Initialize(gameView);
            }
        }

        public float ScalingFactor()
        {
            float density;
            float densityX = this.Resources.DisplayMetrics.Xdpi;
            float densityY = this.Resources.DisplayMetrics.Ydpi;
            if (densityX > densityY)
            {
                density = densityX;
            }
            else
            {
                density = densityY;
            }

            density /= 320;

            // Safeguards against badly reporting density.  10 would be 3200 DPI, which is much higher than what exists today.
            if (density < 1) { density = 1; }
            if (density > 10) { density = 10; }

            return density;
        }

        public override void OnRequestPermissionsResult(int requestCode, string[] permissions, [GeneratedEnum] Android.Content.PM.Permission[] grantResults)
        {
            Xamarin.Essentials.Platform.OnRequestPermissionsResult(requestCode, permissions, grantResults);

            base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
        }

        //
        // Save\load state functions
        //
        public static int GetSavedIntWithDefaultValue(int defaultValue, string filename)
        {
            int ret = defaultValue;

            string folder = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Personal);

            string filePath = Path.Combine(folder, filename);

            if (File.Exists(filePath))
            {
                FileStream file = File.OpenRead(filePath);

                if (file.CanRead)
                {
                    XmlSerializer ser = new XmlSerializer(typeof(int));
                    try
                    {
                        ret = (int)ser.Deserialize(file);
                    }
                    catch { }
                }

                file.Dispose();
                file.Close();
            }

            return ret;
        }

        public static void SaveIntValue(int value, string filename)
        {
            XmlSerializer ser = new XmlSerializer(typeof(int));

            string folder = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Personal);

            string filePath = Path.Combine(folder, filename);

            try
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
                FileStream file = File.Create(filePath);

                ser.Serialize(file, value);

                file.Flush();
                file.Dispose();
                file.Close();
            }
            catch { }
        }
    }
}