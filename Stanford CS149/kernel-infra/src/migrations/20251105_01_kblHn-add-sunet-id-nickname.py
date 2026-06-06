"""
add_sunet_id_nickname
"""

from yoyo import step

__depends__ = {'20250822_01_UtXzl-website-submission'}

steps = [
    step("""
        ALTER TABLE leaderboard.user_info
        ADD COLUMN IF NOT EXISTS sunet_id TEXT;
        ALTER TABLE leaderboard.user_info
        ADD COLUMN IF NOT EXISTS nickname TEXT;
         """
     )
]
