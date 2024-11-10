import click


@click.command()
def run_ui():
    import napari

    viewer = napari.Viewer()
    viewer.window.add_plugin_dock_widget("brainways")
    napari.run()


if __name__ == "__main__":
    run_ui()
