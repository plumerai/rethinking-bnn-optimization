from zookeeper import cli, build_train
from os import path
import click
from bnn_optimization import utils


@cli.command()
@click.option("--tensorboard/--no-tensorboard", default=True)
@build_train(utils.prepare_registry)
def train(build_model, dataset, hparams, output_dir, tensorboard):
    import larq as lq
    from bnn_optimization import utils, data
    import tensorflow as tf

    initial_epoch = utils.get_current_epoch(output_dir)
    model_path = path.join(output_dir, "model")
    callbacks = [utils.ModelCheckpoint(filepath=model_path, save_weights_only=True)]
    if hasattr(hparams, "learning_rate_schedule"):
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(hparams.learning_rate_schedule)
        )
    if tensorboard:
        callbacks.extend(
            [
                tf.keras.callbacks.TensorBoard(
                    log_dir=output_dir,
                    write_graph=False,
                    histogram_freq=0,
                    update_freq=250,
                ),
            ]
        )

    with tf.device("/cpu:0"):
        train_data = dataset.train_data(hparams.batch_size)
        validation_data = dataset.validation_data(hparams.batch_size)

    with utils.get_distribution_scope(hparams.batch_size):
        model = build_model(hparams, **dataset.preprocessing.kwargs)
        model.compile(
            optimizer=hparams.optimizer,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
        )
        lq.models.summary(model)

        if initial_epoch > 0:
            model.load_weights(model_path)
            click.echo(f"Loaded model from epoch {initial_epoch}")

    model.fit(
        train_data,
        epochs=hparams.epochs,
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=validation_data,
        validation_steps=dataset.validation_examples // hparams.batch_size,
        verbose=2 if tensorboard else 1,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    model_name = build_model.__name__
    model.save_weights(path.join(output_dir, f"{model_name}_weights.h5"))


if __name__ == "__main__":
    cli()
