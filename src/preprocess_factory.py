import ast
import inspect
import tensorflow as tf


class preprocess_factory(object):
    # TODO Pre-processing factory
    # Normalize:
    # - Subtract mean
    # - divide by standard deviation
    # -- Fixed, from batch, from image
    # Random crop
    # - specify boundaries
    # Random scale
    # - scale by 0.9 to 1.1 (default)
    # Random flip
    # - Horizontal
    # - vertical

    def __init__(self):
        self._pipe_params = [(self.nop,{})]

    def prep_pipe(self, pipe_params):
        self._pipe_params = pipe_params
    
    def prep_pipe_from_string(self, pipe_params_string):
        pipe_params_parts = pipe_params_string.split(';')
        #TODO: Assert length? Must be a multiple of 2
        pipe_params = list(zip(*[iter(pipe_params_parts)]*2))

        # Convert string names to method references
        pipe_params = [([m_ref for m_name, m_ref in inspect.getmembers(self) if m_name == param_pair[0]][0], ast.literal_eval(param_pair[1])) for param_pair in pipe_params]

        self.prep_pipe(pipe_params)

        return


    def pipe(self, tf_image, tf_labels, *args):
        pipe_params = self._pipe_params
        for process_tuple in pipe_params:
            _process, param_dict = process_tuple
            tf_image, tf_labels, *args = _process(param_dict, tf_image, tf_labels, *args)
        return (tf_image, tf_labels, *args)

    def nop(self, param_dict, tf_image, tf_label, *args):
        # Dummy preprocessing function performing no operation
        pass

        return (tf_image, tf_label, *args)

    def pad_to_size(self, param_dict, tf_image, tf_label, *args):
        # Pad image with zeros on all four sides to produce an output image of size [target_height, target_width]
        # The input image will be approximately centered in the output image.
        #
        # input:
        #   param_dict      dict with the following key-value pairs:
        #                       'target_height': height in pixels of output image (int)
        #                       'target_width': width in pixels of output image (int)
        # 

        target_height = param_dict['height']
        target_width = param_dict['width']

        with tf.name_scope('pad_to_size_' + '{:d}'.format(target_height) + '_' + '{:d}'.format(target_width)):
            tf_image_shape = tf.shape(tf_image)
            image_height = tf.cond(tf.equal(tf.rank(tf_image),4),lambda: tf_image_shape[1], lambda: tf_image_shape[0], name='image_height')
            image_width = tf.cond(tf.equal(tf.rank(tf_image),4),lambda: tf_image_shape[2],lambda: tf_image_shape[1], name='image_width')
            tf_image = tf.image.pad_to_bounding_box(tf_image,
                                                        offset_height=tf.floordiv(target_height-image_height,2),
                                                        offset_width=tf.floordiv(target_width-image_width,2),
                                                        target_height=target_height,
                                                        target_width=target_width)

        return (tf_image, tf_label, *args)

    def crop_pad_to_size(self, param_dict, tf_image, tf_label, *args):
        # Crop or pad image with zeros on all four sides to produce an output image of size [height, width]
        #
        # input:
        #   param_dict      dict with the following key-value pairs:
        #                       'height': height in pixels of output image (int)
        #                       'width': width in pixels of output image (int)
        # 
        
        height = param_dict['height']
        width = param_dict['width']

        with tf.name_scope('crop_pad_to_size_' + '{:d}'.format(height) + '_' + '{:d}'.format(width)):
            tf_image = tf.image.resize_image_with_crop_or_pad(tf_image, target_height=height, target_width=width)
        return (tf_image, tf_label, *args)

    def random_rotation(self, param_dict, tf_image, tf_label, *args):
        # Rotate image by a random angle drawn from a uniform distribution in range [min_angle, max_angle]
        #
        #   param_dict      dict with the following key-value pairs:
        #                       'min_angle': lower limit of random angle in radians (float, default = 0)
        #                       'max_angle': upper limit of random angle in radians (float, default = 2*pi)
        #
        # NOTE: Only works on single images!
        # TODO: Extend to batch of images

        
        min_angle = param_dict.get('min_angle', 0)
        max_angle = param_dict.get('max_angle', 6.28318530718)
        interp_method_string = param_dict.get('interp_method','bilinear')
        if (interp_method_string == 'bilinear'):
            interp_method = 'BILINEAR'
        elif (interp_method_string == 'nearest'):
            interp_method = 'NEAREST'
        else:
            raise ValueError('Unknown interpolation method (' + interp_method + '). Expected on of the following \'bilinear\' (default) or \'nearest\'')
        with tf.name_scope('random_rotation_'+ '{:5.3f}'.format(min_angle) + '_' + '{:5.3f}'.format(max_angle) + ''):
            rotations = tf.random.uniform([1,],minval=min_angle,maxval=max_angle)
            tf_image = tf.contrib.image.rotate(tf_image, angles=rotations, interpolation=interp_method)

        return (tf_image, tf_label, *args)

    def resize(self, param_dict, tf_image, tf_label, *args):
        # Resize image to height and width
        #
        #   param_dict      dict with the following key-value pairs:
        #                       'height': height in pixels of output image (int)
        #                       'width': width in pixels of output image (int)
        #                       'resize_method': method used for interpolating between pixels.
        #                                        Must be one of the following strings:
        #                                        'bilinear' (default), 'nearest', 'bicubic' or 'area'
        #
        # See also: tf.image.resize_images()

        height = param_dict['height']
        width = param_dict['width']
        resize_method_string = param_dict.get('resize_method', 'bilinear')
        if (resize_method_string == 'bilinear'):
            resize_method = tf.image.ResizeMethod.BILINEAR
        elif (resize_method_string == 'nearest'):
            resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif (resize_method_string == 'bicubic'):
            resize_method = tf.image.ResizeMethod.BICUBIC
        elif (resize_method_string == 'area'):
            resize_method = tf.image.ResizeMethod.AREA
        else:
            raise ValueError('Unknown resize method (' + resize_method_string + '). Expected on of the following \'bilinear\' (default), \'nearest\', \'bicubic\' or \'area\'')

        with tf.name_scope('resize_' + '{:d}'.format(height) + '_' + '{:d}'.format(width)):
            tf_image = tf.image.resize_images(tf_image,
                                            size=[height, width],
                                            method=resize_method
                                            )

        return (tf_image, tf_label, *args)

    def scale_values(self, param_dict, tf_image, tf_label, *args):
        # Scale image values from between [min_in;max_in] to [min_out;max_out]
        #
        #   param_dict      dict with the following key-value pairs:
        #                       'min_in': minimum value of input
        #                       'max_in': maximum value of input
        #                       'min_out': minimum value of output
        #                       'max_out': maximum value of output
        #                       'truncate': truncate values outside [min_in;max_in] to min_in or max_in before scaling. Default = True.

        min_in = param_dict['min_in']
        max_in = param_dict['max_in']
        min_out = param_dict['min_out']
        max_out = param_dict['max_out']
        truncate = param_dict.get('truncate', True)

        with tf.name_scope('scale_values_' + '{:.3f}'.format(min_in) + '_' + '{:.3f}'.format(max_in) + '_' + '{:.3f}'.format(min_out) + '_' + '{:.3f}'.format(max_out)):
            tf_image = tf.to_float(tf_image)

            # Truncate values
            if (truncate):
                tf_image = tf.math.minimum(tf_image,max_in)
                tf_image = tf.math.maximum(tf_image,min_in)
            
            # Scale to range [0,1]
            tf_image = (tf_image - min_in)/(max_in - min_in)
            # Scale to range [min_out, max_out]
            tf_image = tf_image*(max_out - min_out) + min_out

            #TODO: Convert tf_image back to original dtype?

        return (tf_image, tf_label, *args)