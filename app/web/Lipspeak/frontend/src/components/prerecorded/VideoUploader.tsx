/** VideoUploader.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video upload component for pre-recorded speech analysis.
 * @date: 10 June 2026
 * @returns: Video uploader component.
 *
 */


// Client Component
"use client";


// Imports
import { UploadCloud } from "lucide-react";


// Props Interface
interface VideoUploaderProps {
  onVideoSelect: (
    file: File,
  ) => void;

  selectedFile: File | null;
}


// Video Uploader Component
export function VideoUploader({
  onVideoSelect,
  selectedFile,
}: Readonly<VideoUploaderProps>) {
  /* ---------------------------------------------------------------------- */
  /*                              Handlers                                  */
  /* ---------------------------------------------------------------------- */

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file =
      event.target.files?.[0];

    if (!file) return;

    onVideoSelect(file);
  };

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
      <div className="mb-6">
        <h2
          className="
            text-2xl
            font-bold
          "
        >
          Upload Video
        </h2>

        <p
          className="
            mt-2
            text-sm
            text-muted-foreground
          "
        >
          Upload a pre-recorded video
          for AI-powered speech analysis.
        </p>
      </div>

      <label
        htmlFor="video-upload"
        className="
          flex
          cursor-pointer
          flex-col
          items-center
          justify-center
          rounded-3xl
          border-2
          border-dashed
          border-primary/30
          px-6
          py-14
          text-center
          transition-all
          duration-300
          hover:border-primary
          hover:bg-primary/5
        "
      >
        <UploadCloud
          className="
            h-14
            w-14
            text-primary
          "
        />

        <h3
          className="
            mt-4
            text-lg
            font-semibold
          "
        >
          Select a Video
        </h3>

        <p
          className="
            mt-2
            text-sm
            text-muted-foreground
          "
        >
          MP4, MOV or WebM supported.
        </p>

        {selectedFile && (
          <div
            className="
              mt-6
              rounded-xl
              bg-primary/10
              px-4
              py-3
              text-sm
              font-medium
              text-primary
            "
          >
            {selectedFile.name}
          </div>
        )}
      </label>

      <input
        id="video-upload"
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}