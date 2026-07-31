import { Response } from 'express';

export function sendSuccess<T>(
  res: Response,
  data: T,
  message = 'Success',
  statusCode = 200
): Response {
  return res.status(statusCode).json({ success: true, message, data });
}

export function sendError(
  res: Response,
  message: string,
  statusCode = 400,
  errors?: unknown
): Response {
  return res.status(statusCode).json({
    success: false,
    message,
    ...(errors !== undefined ? { errors } : {}),
  });
}

export function sendCreated<T>(res: Response, data: T, message = 'Created'): Response {
  return sendSuccess(res, data, message, 201);
}
