from psychopy import core, event, visual, sound


def check_escape(window):
    if len(event.getKeys('escape')) > 0:
        window.close()
        core.quit()
    event.clearEvents()


def beep(window, duration=0.3, note='A'):
    s = sound.Sound(note,
                    secs=duration)
    s.play()

    window.flip()


def fixation(window):
    return visual.Circle(window,
                         radius=0.01,
                         units='height',
                         fillColor=(1., 1., 1.),
                         fillColorSpace='rgb',
                         pos=(0, 0),
                         )


def resting_state(window, duration, eyes_open=True, text=None, duration_instruction=5, background_color=(0., 0., 0.),
                  trig_start=None, trig_end=None, parport=None, letter_height=0.02):
    if text is None:
        if eyes_open:
            print("--- Resting state eyes open ---")

            text = 'Please focus your eyes on the dot and relax.'
        else:
            print("--- Resting state eyes closed ---")
            text = 'When you hear a beep, please close your eyes and relax. You will hear another beep when you can ' \
                   'open your eyes again.'

    win_color_before = window.color
    instruction = visual.TextBox2(win=window,
                                  text=text,
                                  color=(1., 1., 1.),
                                  letterHeight=letter_height,
                                  colorSpace='rgb',
                                  pos=(0., 0.),
                                  alignment='center',
                                  lineBreaking='uax14',
                                  borderColorSpace='rgb',
                                  fillColorSpace='rgb')
    fix = fixation(window)

    change_bg_color(window, background_color)
    instruction.draw()

    window.flip()
    core.wait(duration_instruction)
    if not eyes_open:
        beep(window=window)
    fix.draw()

    window.flip()
    send_trig_if_parport(trigger=trig_start, parport=parport)
    core.wait(duration)
    send_trig_if_parport(trigger=trig_end, parport=parport)
    if not eyes_open:
        beep(window=window)
    change_bg_color(window, win_color_before)


def change_bg_color(window, color):
    """Change the background color of a window.
    CAREFUL: This operation takes two window.flips()!!(see https://www.psychopy.org/api/visual/window.html,
    under `property color`)

    :param window: psychopy.window which background will be changed
    :param color: the new background color of the window, see https://www.psychopy.org/general/colours.html

    :returns: nothing
    """

    window.setColor(color, colorSpace='rgb')
    window.flip()
    window.flip()


def send_trig_if_parport(trigger, parport, wait=0.001):
    if parport is not None:
        if trigger is not None:
            parport.setData(trigger)
            core.wait(wait)
            parport.setData(0)


def paginated_text_with_pics(window, texts: list, pictures: list, bg_color=(0.2, 0.2, 0.2), text_color=(1.,1.,1.),
                             letter_height=None):
    if len(texts) != len(pictures):
        raise ValueError("texts and pictures mus be of the same length.")

    textbox_press_arrow = visual.TextBox2(win=window,
                                      text="Press the right arrow to continue.",
                                      letterHeight=0.05,
                                      bold=False,
                                      color=text_color,
                                      colorSpace='rgb',
                                      pos=(0, -0.8),
                                      anchor='center',
                                      alignment='center')

    textbox_with_pic = visual.TextBox2(win=window,
                                       text="",
                                       letterHeight=letter_height,
                                       color=text_color,
                                       colorSpace='rgb',
                                       pos=(0., 0.),
                                       units="norm",
                                       anchor='left',
                                       alignment='left',
                                       padding=0.05)
    image_stim = visual.ImageStim(win=window,
                                  image=None,
                                  pos=(0., 0.),
                                  units="norm",
                                  anchor='right',
                                  size=(0.95, 0.95))

    textbox_right_arrow = visual.TextBox2(win=window,
                                          text=">",
                                          letterHeight=0.2,
                                          bold=True,
                                          color=text_color,
                                          colorSpace='rgb',
                                          pos=(0.9, -0.8),
                                          anchor='right',
                                          alignment='center')

    textbox_left_arrow = visual.TextBox2(win=window,
                                         text="<",
                                         letterHeight=0.2,
                                         bold=True,
                                         color=text_color,
                                         colorSpace='rgb',
                                         pos=(-0.9, -0.8),
                                         units="norm",
                                         anchor='left',
                                         alignment='center')

    change_bg_color(window, bg_color)
    textbox_press_arrow.draw()
    page_nr = 0
    while page_nr < len(texts):
        textbox_with_pic.setText(texts[page_nr])
        textbox_with_pic.draw()
        image_stim.setImage(pictures[page_nr])
        image_stim.draw()
        if page_nr > 0:
            textbox_left_arrow.draw()
        if page_nr != len(texts) - 1:
            textbox_right_arrow.draw()
        window.flip()

        print(f"Showing page {page_nr}, waiting for key press...")
        pressed = event.waitKeys(keyList=['left', 'right', 'escape', 'space'])
        if pressed[-1] == 'right' and page_nr != len(texts) - 1:
            event.clearEvents()
            page_nr += 1
        elif pressed[-1] == 'left':
            event.clearEvents()
            if page_nr > 0:
                page_nr -= 1
        elif pressed[-1] == 'space' and page_nr == len(texts) - 1:
            return
        elif pressed[-1] == 'escape':
            window.close()
            core.quit()






if __name__ == "__main__":
    from psychopy import visual, core, event, gui

    # present dialog to collect info
    info = {'participant': 'x',
            'session': 1,
            'experimenter': '',
            'language': 'en',
            'participant_screen': 1,
            }
    dlg = gui.DlgFromDict(info, sortKeys=False)
    if not dlg.OK:
        core.quit()

    window = visual.Window(size=(1920, 1080),
                           monitor='testMonitor',
                           fullscr=True,
                           units="norm",
                           color=(0.2, 0.2, 0.2),
                           colorSpace='rgb',
                           screen=info['participant_screen'],
                           )

    pics = ["../../experiments/figs/explanation_1.png",
            "../../experiments/figs/explanation_2.png",
            "../../experiments/figs/explanation_3.png",
            "../../experiments/figs/explanation_4.png",
            "../../experiments/figs/explanation_5.png"]

    paginated_text_with_pics(window,
                             texts=['text page 1', "text page 2", "text page 3", "text page 4", "text page 5"],
                             pictures=pics,
                             letter_height=None)

    print('done')
    core.quit()
