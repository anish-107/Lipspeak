/** video.service.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video related API service functions.
 * @date: 09 June 2026
 * @returns: Video API methods.
 *
 */


// Imports
import { apiClient } from "@/lib/api-client";
import type {
  Video,
  UploadVideoResponse,
} from "@/types/video.types";


// Video Service
export const videoService = {
  /* ------------------------------------------------------------------------ */
  /*                            Get User Videos                               */
  /* ------------------------------------------------------------------------ */

  async getVideos(): Promise<Video[]> {
    const response = await apiClient.get<Video[]>(
      "/videos",
    );

    return response.data;
  },


  /* ------------------------------------------------------------------------ */
  /*                             Get Single Video                             */
  /* ------------------------------------------------------------------------ */

  async getVideoById(
    videoId: string,
  ): Promise<Video> {
    const response = await apiClient.get<Video>(
      `/videos/${videoId}`,
    );

    return response.data;
  },


  /* ------------------------------------------------------------------------ */
  /*                              Upload Video                                */
  /* ------------------------------------------------------------------------ */

  async uploadVideo(
    file: File,
  ): Promise<UploadVideoResponse> {
    const formData = new FormData();

    formData.append(
      "file",
      file,
    );

    const response =
      await apiClient.post<UploadVideoResponse>(
        "/videos/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );

    return response.data;
  },


  /* ------------------------------------------------------------------------ */
  /*                             Delete Video                                 */
  /* ------------------------------------------------------------------------ */

  async deleteVideo(
    videoId: number,
  ): Promise<void> {
    await apiClient.delete(
      `/videos/${videoId}`,
    );
  },
};