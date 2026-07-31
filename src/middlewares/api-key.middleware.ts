import { Request, Response, NextFunction } from 'express';
import { config } from '../config';

export function requireApiKey(req: Request, res: Response, next: NextFunction): void {
  const key = req.headers['x-api-key'];
  if (!key || key !== config.app.internalApiKey) {
    res.status(401).json({ success: false, message: 'Invalid or missing API key' });
    return;
  }
  next();
}
