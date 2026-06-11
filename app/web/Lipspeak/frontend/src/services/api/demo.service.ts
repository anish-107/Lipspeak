/**
 * demo.service.ts
 * @authors: Anish Kumar, Bidipta Barua,
 * Dibyasmita Hati, Arpan Haldar
 * @description: Homepage demo API service.
 * @date: 11 June 2026
 * @returns: Demo API methods.
 *
 */


// Imports
import {
  apiClient,
} from "@/lib/api-client";


// Demo Service
export const demoService = {
  async transcribe(
    file: File,
  ) {
    const formData =
      new FormData();

    formData.append(
      "file",
      file,
    );

    const response =
      await apiClient.post(
        "/demo/transcribe",
        formData,
        {
          headers: {
            "Content-Type":
              "multipart/form-data",
          },
        },
      );

    return response.data;
  },
};