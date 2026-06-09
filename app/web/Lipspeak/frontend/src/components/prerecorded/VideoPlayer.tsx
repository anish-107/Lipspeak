/** VideoPlayer.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video preview player for uploaded pre-recorded videos.
 * @date: 10 June 2026
 * @returns: Video player component.
 *
 */


// Props Interface
interface VideoPlayerProps {
  videoUrl: string;
}


// Video Player Component
export function VideoPlayer({
  videoUrl,
}: Readonly<VideoPlayerProps>) {
  /* ---------------------------------------------------------------------- */
  /*                             Empty State                                */
  /* ---------------------------------------------------------------------- */

  if (!videoUrl) {
    return (
      <div
        className="
          glass-card
          ai-border
          rounded-3xl
          p-6
        "
      >
        <h2
          className="
            text-2xl
            font-bold
          "
        >
          Video Preview
        </h2>

        <div
          className="
            mt-6
            flex
            h-72
            items-center
            justify-center
            rounded-2xl
            border
            border-dashed
            border-border
          "
        >
          <p className="text-muted-foreground">
            Upload a video to preview it.
          </p>
        </div>
      </div>
    );
  }

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <div
      className="
        glass-card
        ai-border
        rounded-3xl
        p-6
      "
    >
      <h2
        className="
          text-2xl
          font-bold
        "
      >
        Video Preview
      </h2>

      <div className="mt-6">
        <video
          controls
          preload="metadata"
          className="
            w-full
            rounded-2xl
            border
            border-border
            object-cover
          "
        >
          <source
            src={videoUrl}
            type="video/mp4"
          />

          Your browser does not support
          video playback.
        </video>
      </div>
    </div>
  );
}