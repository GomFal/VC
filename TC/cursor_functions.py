import pyautogui
import time
import pygetwindow as gw

pyautogui.FAILSAFE = False

def scroll_based_on_cursor_position(x, y, screen_width, screen_height, scroll_delay, last_scroll_time, scroll_confirm_time=3):
    """
    Desplazarse hacia arriba o hacia abajo según la posición del cursor.
    
    :param x: Posición x actual del cursor
    :param y: Posición y actual del cursor
    :param screen_width: Ancho de la pantalla
    :param screen_height: Altura de la pantalla
    :param scroll_delay: Retraso antes de desplazarse
    :param last_scroll_time: Última vez que se realizó la acción de desplazamiento
    :param scroll_confirm_time: Tiempo para confirmar la acción de desplazamiento
    :return: Tiempo actualizado de la última acción de desplazamiento
    """
    current_time = time.time()
    active_window = gw.getActiveWindow()
    if active_window:
        active_window.activate()
    
    if x > screen_width * 0.95 and y < screen_height * 0.05:
        if current_time - last_scroll_time > scroll_delay:
            time.sleep(scroll_confirm_time)  # Esperar el tiempo de confirmación
            if x > screen_width * 0.95 and y < screen_height * 0.05:  # Verificar la posición nuevamente
                pyautogui.moveTo(screen_width // 2, screen_height // 2)  # Mover el cursor al centro de la página
                for _ in range(15):  # Desplazamiento suave hacia arriba
                    pyautogui.scroll(10)
                    time.sleep(0.05)
                last_scroll_time = current_time
    elif x > screen_width * 0.95 and y > screen_height * 0.95:
        if current_time - last_scroll_time > scroll_delay:
            time.sleep(scroll_confirm_time)  # Esperar el tiempo de confirmación
            if x > screen_width * 0.95 and y > screen_height * 0.95:  # Verificar la posición nuevamente
                pyautogui.moveTo(screen_width // 2, screen_height // 2)  # Mover el cursor al centro de la página
                for _ in range(15):  # Desplazamiento suave hacia abajo
                    pyautogui.scroll(-10)
                    time.sleep(0.05)
                last_scroll_time = current_time
    return last_scroll_time