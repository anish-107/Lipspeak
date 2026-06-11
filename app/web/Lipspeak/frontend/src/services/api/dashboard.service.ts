/** dashboard.service.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard API service.
 * @date: 11 June 2026
 * @returns: Dashboard API operations.
 *
 */


// Imports
import {
  apiClient,
} from "@/lib/api-client";


// Dashboard Service
export const dashboardService = {
  async getOverview() {
    const response =
      await apiClient.get(
        "/dashboard/overview",
      );

    return response.data;
  },
};