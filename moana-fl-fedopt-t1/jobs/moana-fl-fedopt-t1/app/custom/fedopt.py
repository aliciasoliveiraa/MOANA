import time

import tensorflow as tf

from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.security.logging import secure_format_exception


class TFFedOptModelShareableGenerator(FullModelShareableGenerator):
    def __init__(
        self,
        optimizer_args: dict = None,
        lr_scheduler_args: dict = None,
        source_model="model",
        device=None,
        image_size=256,
        num_contrast=4,
    ):
        """Implement the FedOpt algorithm.

        The algorithm is proposed in Reddi, Sashank, et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).
        This SharableGenerator will update the global model using the specified
        TensorFlow optimizer and learning rate scheduler.
        Note: This class will use FedOpt to optimize the global trainable parameters (i.e. `self.model.trainable_variables`)
        but use FedAvg to update any other layers such as batch norm statistics.

        Args:
            optimizer_args: dictionary of optimizer arguments, e.g.
                {'path': 'tf.keras.optimizers.SGD', 'args': {'learning_rate': 1.0}} (default).
            lr_scheduler_args: dictionary of server-side learning rate scheduler arguments, e.g.
                {'path': 'tf.keras.optimizers.schedules.CosineDecay', 'args': {'decay_steps': 100, 'alpha': 0.0}} (default: None).
            source_model: either a valid tf.keras model object or a component ID of a tf.keras model object
            device: specify the device to run server-side optimization, e.g. "cpu" or "gpu:0"
                (will default to GPU if available and no device is specified).

        Raises:
            TypeError: when any of input arguments does not have correct type
        """
        super().__init__()
        if not optimizer_args:
            self.logger("No optimizer_args provided. Using FedOpt with SGD and learning_rate 1.0")
            optimizer_args = {"name": "SGD", "args": {"learning_rate": 1.0}}

        if not isinstance(optimizer_args, dict):
            raise TypeError(
                "optimizer_args must be a dict of format, e.g. {'path': 'tf.keras.optimizers.SGD', 'args': {'learning_rate': 1.0}}."
            )
        if lr_scheduler_args is not None:
            if not isinstance(lr_scheduler_args, dict):
                raise TypeError(
                    "lr_scheduler_args must be a dict of format, e.g. "
                    "{'path': 'tf.keras.optimizers.schedules.CosineDecay', 'args': {'decay_steps': 100, 'alpha': 0.0}}."
                )
        self.source_model = source_model
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        if device is None:
            self.device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
        else:
            self.device = device
        self.optimizer_name = None
        self.lr_scheduler_name = None
        self.image_size = image_size
        self.num_contrast = num_contrast

    def _get_component_name(self, component_args):
        if component_args is not None:
            name = component_args.get("path", None)
            if name is None:
                name = component_args.get("name", None)
            return name
        else:
            return None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            # Initialize the optimizer with current global model params
            engine = fl_ctx.get_engine()

            if isinstance(self.source_model, str):
                self.model = engine.get_component(self.source_model)
            else:
                self.model = self.source_model

            if self.model is None:
                self.system_panic(
                    "Model is not available",
                    fl_ctx,
                )
                return
            elif not isinstance(self.model, tf.keras.Model):
                self.system_panic(
                    f"Expected model to be a tf.keras.Model but got {type(self.model)}",
                    fl_ctx,
                )
                return
            else:
                print("server model", self.model)

            self.model.compile(run_eagerly=True)

            for i, layer in enumerate(self.model.layers):
                layer._name = 'layer_' + str(i)
            '''
            for layer in self.model.layers:
                print('LAYER NAME FL', layer.name)
            '''
            # Explicitly build the model to create weights
            input_shape = [(None, self.image_size, self.image_size, 1)]
            self.model.build(input_shape=input_shape * self.num_contrast)
            
            # set up optimizer
            try:
                # use provided or default optimizer arguments and add the model parameters
                optimizer_class = getattr(tf.keras.optimizers, self.optimizer_args['name'])
                #print('optimizer_class', optimizer_class)
                self.optimizer = optimizer_class(**self.optimizer_args['args'])
                #print('self.optimizer', self.optimizer)
                # get optimizer name for log
                self.optimizer_name = self._get_component_name(self.optimizer_args)
                #print('self.optimizer_name', self.optimizer_name)
            except Exception as e:
                self.system_panic(
                    f"Exception while parsing `optimizer_args`({self.optimizer_args}): {secure_format_exception(e)}",
                    fl_ctx,
                )
                return

            # set up lr scheduler
            if self.lr_scheduler_args is not None:
                try:
                    self.lr_scheduler_name = self._get_component_name(self.lr_scheduler_args)
                    #print('self.lr_scheduler_name', self.lr_scheduler_name)
                    lr_schedule_class = getattr(tf.keras.optimizers.schedules, self.lr_scheduler_args['name'])
                    #print('lr_schedule_class', lr_schedule_class)
                    self.lr_scheduler = lr_schedule_class(**self.lr_scheduler_args['args'])
                    #print('self.lr_scheduler', self.lr_scheduler)
                    self.optimizer.learning_rate = self.lr_scheduler
                except Exception as e:
                    self.system_panic(
                        f"Exception while parsing `lr_scheduler_args`({self.lr_scheduler_args}): {secure_format_exception(e)}",
                        fl_ctx,
                    )
                    return

    def server_update(self, model_diff):
        """Updates the global model using the specified optimizer.

        Args:
            model_diff: the aggregated model differences from clients.

        Returns:
            The updated TensorFlow model weights and the list of updated parameters names.

        """
        self.model.trainable = True

        updated_params = []
        
        with tf.GradientTape() as tape:
            print('variables', self.model.variables)
            print('trainable_variables', self.model.trainable_variables)
            print('model_diff', model_diff)
            print('model_diff.items', model_diff.items())
            counter_dict = {} 
            
            for var in self.model.trainable_variables:
                print('var.name', var.name)
                layer_name = var.name.split("/")[0]
                if layer_name not in counter_dict:
                    counter_dict[layer_name] = 0
                
                var_name = f"{layer_name}_nvf_{counter_dict[layer_name]}"
                print('var_name', var_name)

                if var_name in model_diff:
                    diff = model_diff[var_name]
                    print('diff', diff)
                    
                    diff_tensor = tf.convert_to_tensor(-1.0 * diff, dtype=var.dtype)
                    print('diff_tensor', diff_tensor)
                    
                    var.assign_add(diff_tensor)
                    print('var.assign_add(diff_tensor)', var.numpy())
                    updated_params.append(var.name)
                
                counter_dict[layer_name] += 1
            
            updated_vars = [var for var in self.model.trainable_variables if f"{var.name.split('/')[0]}_nvf_{counter_dict[var.name.split('/')[0]] - 1}" in model_diff]
            print('updated_vars', updated_vars)
            gradients = tape.gradient(self.model.trainable_variables, updated_vars)
            print('gradients', gradients)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return self.model.get_weights(), updated_params


    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        """Convert Shareable to Learnable while doing a FedOpt update step.

        Supporting data_kind == DataKind.WEIGHT_DIFF

        Args:
            shareable (Shareable): Shareable to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Model: Updated global ModelLearnable.
        """
        # check types
        dxo = from_shareable(shareable)

        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            self.system_panic(
                "FedOpt is only implemented for " "data_kind == DataKind.WEIGHT_DIFF",
                fl_ctx,
            )
            return Learnable()

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            self.system_panic(
                f"FedOpt is not implemented for shareable processed by {processed_algorithm}",
                fl_ctx,
            )
            return Learnable()

        model_diff = dxo.data

        start = time.time()
        weights, updated_params = self.server_update(model_diff)
        secs = time.time() - start

        print('weights', weights)
        print('updated_params', updated_params)
        
        # Convert model weights to dictionary
        weights_dict = {}
        for var, weight in zip(self.model.trainable_variables, weights):
            weights_dict[var.name] = weight
        
        # update unnamed parameters such as batch norm layers if there are any using the averaged update
        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
            return base_model

        base_model_weights = base_model[ModelLearnableKey.WEIGHTS]

        n_fedavg = 0
        for key, value in model_diff.items():
            if key not in updated_params:
                print('key', key)
                weights_dict[key] = base_model_weights[key] + value
                print('weights_dict[key]', weights_dict[key])
                n_fedavg += 1
        
        self.log_info(
            fl_ctx,
            f"FedOpt ({self.optimizer_name}, {self.device}) server model update "
            f"round {fl_ctx.get_prop(AppConstants.CURRENT_ROUND)}, "
            f"{self.lr_scheduler_name if self.lr_scheduler_name else ''} "
            f"lr: {self.optimizer.learning_rate.numpy()}, "
            f"fedopt layers: {len(updated_params)}, "
            f"fedavg layers: {n_fedavg}, "
            f"update: {secs} secs.",
        )
        # TODO: write server-side lr to tensorboard

        return make_model_learnable(base_model_weights, dxo.get_meta_props())
