from IPython.core.display import display
import glob
import os
from IPython.display import Video
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

def plot_videos(videos_path='recording', output_file_path='.'):
	stringa = 'ffmpeg -i \"concat:'
	elenco_video = glob.glob(f'{videos_path}/*.mp4')
	if len(elenco_video) == 0:
		print('0 mp4 found in this path')
		return
	elenco_file_temp = []
	for f in elenco_video:
		file = videos_path + '/temp' + str(elenco_video.index(f) + 1) + '.ts'
		os.system('ffmpeg -i ' + f + ' -c copy -bsf:v h264_mp4toannexb -f mpegts ' + file)
		elenco_file_temp.append(file)
	for f in elenco_file_temp:
		stringa += f
		if elenco_file_temp.index(f) != len(elenco_file_temp)-1:
			stringa += '|'
		else:
			stringa += f'\" -c copy -y -bsf:a aac_adtstoasc {output_file_path}'
	os.system(stringa)
	display(Video(output_file_path))

def plot_metrics(n_episodes, total_rewards, number_steps):
    count = len(number_steps)
    _n_episodes = n_episodes[:count].copy()
    _total_rewards = total_rewards[:count].copy()
    _number_steps = number_steps.copy()
    
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=_n_episodes, y=_total_rewards, mode='lines'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=_n_episodes, y=_number_steps, mode='lines'),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Episodes", row=1, col=1)
    fig.update_xaxes(title_text="Episodes", row=1, col=2)
    fig.update_yaxes(title_text="Total rewards", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)

    fig.update_layout(height=600, width=800, showlegend=False)
    fig.show()
