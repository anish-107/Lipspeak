/** auth.store.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Zustand authentication state store.
 * @date: 11 June 2026
 * @returns: Authentication state management.
 *
 */

import { create } from "zustand";

import {
  User,
} from "@/types/auth.types";


interface AuthStore {
  token: string | null;

  user: User | null;

  isLoading: boolean;

  login: (
    token: string,
    user: User,
  ) => void;

  logout: () => void;

  setUser: (
    user: User | null,
  ) => void;

  setToken: (
    token: string | null,
  ) => void;

  setIsLoading: (
    isLoading: boolean,
  ) => void;
}


export const useAuthStore =
  create<AuthStore>(
    (set) => ({
      token: null,

      user: null,

      isLoading: true,

      login: (
        token,
        user,
      ) => {
        localStorage.setItem(
          "access_token",
          token,
        );

        set({
          token,
          user,
        });
      },

      logout: () => {
        localStorage.removeItem(
          "access_token",
        );

        set({
          token: null,
          user: null,
        });
      },

      setUser: (
        user,
      ) =>
        set({
          user,
        }),

      setToken: (
        token,
      ) =>
        set({
          token,
        }),

      setIsLoading: (
        isLoading,
      ) =>
        set({
          isLoading,
        }),
    }),
  );