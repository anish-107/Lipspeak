/**
 * @authors: Anish Kumar, Bidipta Barua,
 * Dibyasmita Hati, Arpan Haldar
 * @description: Authentication hook handling login and signup form logic.
 * @date: 11 June 2026
 * @returns: Authentication handlers and states.
 */

"use client";

import { useState } from "react";

import { useRouter } from "next/navigation";

import { authService }
  from "@/services/api/auth.service";

import { useAuthStore }
  from "@/store/auth.store";

import {
  LoginFormData,
  SignupFormData,
} from "@/types/auth.types";


export function useAuth() {
  const router = useRouter();

  const login = useAuthStore(
    (state) => state.login,
  );

  const [loginLoading, setLoginLoading] =
    useState(false);

  const [signupLoading, setSignupLoading] =
    useState(false);

  const [loginForm, setLoginForm] =
    useState<LoginFormData>({
      username: "",
      password: "",
    });

  const [signupForm, setSignupForm] =
    useState<SignupFormData>({
      username: "",
      name: "",
      password: "",
      confirmPassword: "",
    });

  const handleLoginChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    setLoginForm((previous) => ({
      ...previous,
      [event.target.name]:
        event.target.value,
    }));
  };

  const handleSignupChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    setSignupForm((previous) => ({
      ...previous,
      [event.target.name]:
        event.target.value,
    }));
  };

  const handleLoginSubmit = async (
    event: React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();

    try {
      setLoginLoading(true);

      const response =
        await authService.login({
          username:
            loginForm.username,
          password:
            loginForm.password,
        });

      const user =
        await authService.getCurrentUser();

      login(
        response.access_token,
        user,
      );

      router.push(
        "/dashboard",
      );

    } catch (error) {
      console.error(
        "Login failed:",
        error,
      );
    } finally {
      setLoginLoading(false);
    }
  };

  const handleSignupSubmit = async (
    event: React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();

    try {
      setSignupLoading(true);

      if (
        signupForm.password !==
        signupForm.confirmPassword
      ) {
        throw new Error(
          "Passwords do not match.",
        );
      }

      await authService.signup({
        username: signupForm.username,
        name: signupForm.name,
        password: signupForm.password,
      });

      router.push(
        "/login",
      );

    } catch (error) {
      console.error(
        "Signup failed:",
        error,
      );
    } finally {
      setSignupLoading(false);
    }
  };

  return {
    loginForm,

    signupForm,

    loginLoading,

    signupLoading,

    handleLoginChange,

    handleSignupChange,

    handleLoginSubmit,

    handleSignupSubmit,
  };
}