'use client';

import { Play, Download, Share2, Clock, ThumbsUp } from 'lucide-react';

interface Clip {
  id: string;
  start_time: number;
  end_time: number;
  duration: number;
  caption: string;
  score: number;
  video_url?: string;
  thumbnail_url?: string;
}

interface ClipGalleryProps {
  clips: Clip[];
}

export default function ClipGallery({ clips }: ClipGalleryProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <div className="p-2 bg-green-500/20 rounded-lg mr-3">
            <Play className="h-5 w-5 text-green-400" />
          </div>
          <h2 className="text-xl font-bold">Generated Clips</h2>
        </div>
        <div className="text-sm text-gray-400">
          {clips.length} clips generated
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {clips.map((clip, index) => (
          <div
            key={clip.id}
            className="bg-gray-900 rounded-xl border border-gray-700 overflow-hidden hover:border-gray-600 transition-all"
          >
            {/* Clip Thumbnail */}
            <div className="relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="inline-flex p-4 bg-black/50 rounded-full mb-4">
                    <Play className="h-8 w-8 text-white" />
                  </div>
                  <p className="text-sm text-gray-300">Clip Preview</p>
                </div>
              </div>
              <div className="absolute top-4 left-4">
                <span className="px-3 py-1 bg-black/70 rounded-full text-sm font-medium">
                  #{index + 1}
                </span>
              </div>
              <div className="absolute top-4 right-4 flex items-center space-x-2">
                <span className="px-2 py-1 bg-green-500/20 rounded text-xs text-green-400">
                  {formatTime(clip.duration)}
                </span>
                <span className="px-2 py-1 bg-blue-500/20 rounded text-xs text-blue-400">
                  {(clip.score * 100).toFixed(0)}% match
                </span>
              </div>
            </div>
            
            {/* Clip Info */}
            <div className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <p className="font-medium mb-1">{clip.caption}</p>
                  <p className="text-sm text-gray-400 flex items-center">
                    <Clock className="h-3 w-3 mr-1" />
                    {formatTime(clip.start_time)} - {formatTime(clip.end_time)}
                  </p>
                </div>
                <ThumbsUp className="h-5 w-5 text-yellow-500" />
              </div>
              
              {/* Action Buttons */}
              <div className="flex space-x-2">
                <button className="flex-1 flex items-center justify-center space-x-2 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition">
                  <Play className="h-4 w-4" />
                  <span>Preview</span>
                </button>
                <button className="flex-1 flex items-center justify-center space-x-2 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
                  <Download className="h-4 w-4" />
                  <span>Download</span>
                </button>
                <button className="flex-1 flex items-center justify-center space-x-2 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
                  <Share2 className="h-4 w-4" />
                  <span>Share</span>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Batch Actions */}
      <div className="mt-6 pt-6 border-t border-gray-700">
        <div className="flex justify-between items-center">
          <div className="text-sm text-gray-400">
            Select clips to perform batch actions
          </div>
          <div className="flex space-x-3">
            <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
              Download All
            </button>
            <button className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg font-medium hover:opacity-90 transition">
              Export to TikTok
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
