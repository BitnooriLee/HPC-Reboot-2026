"""
Simple visual leaderboard HTML generator with tabbed interface
"""
from typing import List
from datetime import timezone, timedelta
from libkernelbot.leaderboard_db import LeaderboardRankedEntry

# PST is UTC-8
PST = timezone(timedelta(hours=-8))


def generate_simple_html(leaderboard_data: List[dict]) -> str:
    """
    Generate a tabbed HTML page displaying leaderboards.
    
    Args:
        leaderboard_data: List of dicts with keys: name, gpu_type, submissions
    
    Returns:
        HTML string
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Leaderboards</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: #1a1a1a;
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 30px;
        }
        
        .tabs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .tab {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.2s ease;
            border: 2px solid transparent;
        }
        
        .tab:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        
        .tab.active {
            border-color: #4a9eff;
            box-shadow: 0 4px 12px rgba(74, 158, 255, 0.2);
        }
        
        .tab-header {
            display: flex;
            align-items: center;
            margin-bottom: 16px;
        }
        
        .tab-icon {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 10px;
        }
        
        .tab-title {
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
            font-family: 'Courier New', monospace;
        }
        
        
        .tab-entries {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .entry {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
        }
        
        .entry-left {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .entry-user {
            color: #333;
            font-size: 14px;
        }
        
        .entry-medal {
            font-size: 16px;
        }
        
        .entry-score {
            color: #666;
            font-size: 14px;
            font-family: 'Courier New', monospace;
        }
        
        .leaderboard-full {
            background: white;
            border-radius: 12px;
            padding: 32px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: none;
        }
        
        .leaderboard-full.active {
            display: block;
        }
        
        .leaderboard-header {
            margin-bottom: 24px;
        }
        
        .leaderboard-title {
            font-size: 24px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 8px;
        }
        
        .leaderboard-info {
            color: #666;
            font-size: 14px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        thead {
            border-bottom: 2px solid #e5e5e5;
        }
        
        th {
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: #666;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 16px;
            border-bottom: 1px solid #f0f0f0;
            color: #333;
            font-size: 14px;
        }
        
        tbody tr:hover {
            background-color: #f9f9f9;
        }
        
        .rank {
            font-weight: 600;
            color: #666;
            font-size: 16px;
        }
        
        .medal {
            font-size: 18px;
            margin-left: 8px;
        }
        
        .score {
            font-family: 'Courier New', monospace;
            color: #1a1a1a;
        }
        
        .no-submissions {
            color: #999;
            font-style: italic;
            padding: 40px;
            text-align: center;
        }
        
        .back-button {
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: #f0f0f0;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            color: #333;
            transition: background 0.2s;
        }
        
        .back-button:hover {
            background: #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CS149 Leaderboards</h1>
        
        <div class="tabs" id="tabs">
"""
    
    # Color scheme for tabs
    colors = [
        '#4a9eff',  # blue
        '#9b6bff',  # purple
        '#2dd4bf',  # teal
        '#ec4899',  # pink
        '#f59e0b',  # orange
        '#10b981',  # green
    ]
    
    # Generate tab previews
    for idx, lb in enumerate(leaderboard_data):
        name = lb['name']
        gpu_type = lb['gpu_type']
        submissions = lb['submissions']
        color = colors[idx % len(colors)]

        html += f"""
            <div class="tab" onclick="showLeaderboard({idx})" id="tab-{idx}">
                <div class="tab-header">
                    <div class="tab-icon" style="background-color: {color};"></div>
                    <div class="tab-title">{name}</div>
                </div>
                <div class="tab-entries">
"""
        
        # Show top 3 entries
        medals = ['🥇', '🥈', '🥉']
        top_3 = submissions[:3] if submissions else []
        
        if top_3:
            for i, sub in enumerate(top_3):
                user = sub['user_name']
                score = sub['submission_score']
                medal = medals[i] if i < len(medals) else ''
                
                score_ms = score * 1000  # Convert seconds to milliseconds
                html += f"""
                    <div class="entry">
                        <div class="entry-left">
                            <span class="entry-user">{user}</span>
                            <span class="entry-medal">{medal}</span>
                        </div>
                        <span class="entry-score">{score_ms:.3f}ms</span>
                    </div>
"""
        else:
            html += """
                    <div class="entry">
                        <span style="color: #999; font-size: 13px;">No submissions yet</span>
                    </div>
"""
        
        html += """
                </div>
            </div>
"""
    
    html += """
        </div>
        
"""
    
    # Generate full leaderboard views
    for idx, lb in enumerate(leaderboard_data):
        name = lb['name']
        gpu_type = lb['gpu_type']
        submissions = lb['submissions']
        
        html += f"""
        <div class="leaderboard-full" id="leaderboard-{idx}">
            <button class="back-button" onclick="showTabs()">← Back to all leaderboards</button>
            <div class="leaderboard-header">
                <div class="leaderboard-title">{name}</div>
                <div class="leaderboard-info">{gpu_type}</div>
            </div>
"""
        
        if not submissions:
            html += '            <p class="no-submissions">No submissions yet</p>\n'
        else:
            html += """
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>User</th>
                        <th>Time</th>
                        <th>Submission Time</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            medals = ['🥇', '🥈', '🥉']
            for sub in submissions:
                rank = sub['rank']
                user = sub['user_name']
                score = sub['submission_score']
                time = sub['submission_time']
                medal = medals[rank - 1] if rank <= 3 else ''
                score_ms = score * 1000  # Convert seconds to milliseconds

                # Convert time to PST
                time_pst = time.astimezone(PST)
                time_formatted = time_pst.strftime('%Y-%m-%d %H:%M:%S PST')

                html += f"""
                    <tr>
                        <td class="rank">{rank}{f'<span class="medal">{medal}</span>' if medal else ''}</td>
                        <td>{user}</td>
                        <td class="score">{score_ms:.3f}ms</td>
                        <td>{time_formatted}</td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
"""
        
        html += """
        </div>
"""
    
    html += """
    </div>
    
    <script>
        function showLeaderboard(index) {
            // Hide tabs
            document.getElementById('tabs').style.display = 'none';
            
            // Hide all leaderboards
            const leaderboards = document.querySelectorAll('.leaderboard-full');
            leaderboards.forEach(lb => lb.classList.remove('active'));
            
            // Show selected leaderboard
            document.getElementById('leaderboard-' + index).classList.add('active');
            
            // Update active tab (for when returning)
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            document.getElementById('tab-' + index).classList.add('active');
        }
        
        function showTabs() {
            // Show tabs
            document.getElementById('tabs').style.display = 'grid';
            
            // Hide all leaderboards
            const leaderboards = document.querySelectorAll('.leaderboard-full');
            leaderboards.forEach(lb => lb.classList.remove('active'));
        }
    </script>
</body>
</html>
"""
    
    return html
