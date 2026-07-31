import winston from 'winston';
import path from 'path';
import DailyRotateFile from 'winston-daily-rotate-file';
import { config } from '../config';

const jsonFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

const consoleFormat = winston.format.combine(
  winston.format.colorize(),
  winston.format.timestamp({ format: 'HH:mm:ss' }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    const metaStr = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';
    return `[${timestamp}] ${level}: ${message}${metaStr}`;
  })
);

export const logger = winston.createLogger({
  level: config.logging.level,
  format: jsonFormat,
  defaultMeta: { service: 'leadryze-ai' },
  transports: [
    new DailyRotateFile({
      filename:     path.join(config.logging.dir, 'ai-error-%DATE%.log'),
      datePattern:  'YYYY-MM-DD',
      level:        'error',
      maxFiles:     '90d',
      zippedArchive: true,
    }),
    new DailyRotateFile({
      filename:     path.join(config.logging.dir, 'ai-combined-%DATE%.log'),
      datePattern:  'YYYY-MM-DD',
      maxFiles:     '90d',
      zippedArchive: true,
    }),
  ],
});

if (config.app.env !== 'production') {
  logger.add(new winston.transports.Console({ format: consoleFormat }));
}
