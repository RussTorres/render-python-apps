#!/usr/bin/env python
'''
define a transformation to register the overlapping portion
    of two aligned subvolumes
'''
import numpy
import marshmallow as mm

import renderapi
from ..module.render_module import (
    RenderModule, RenderParameters, OptionList)

example_parameters = {
    "render": {
        "host": "em-131fs",
        "port": 8080,
        "owner": "testuser",
        "project": "test",
        "client_scripts": ""
    },
    "stack_a": "PARENTSTACK",
    "stack_b": "CHILDSTACK",
    "transform_type": "RIGID",
    "pool_size": 12
}


class RegisterSubvolumeModule(RenderModule):
    transform_classes = {'TRANSLATION': renderapi.transform.TranslationModel,
                         'RIGID': renderapi.transform.RigidModel,
                         'SIMILARITY': renderapi.transform.SimilarityModel,
                         'AFFINE': renderapi.transform.AffineModel}

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    def run(self):
        a = self.args['stack_a']
        b = self.args['stack_b']
        r = self.render
        # only interested in registering zs in both stacks
        acoord = numpy.empty([0, 2])
        bcoord = numpy.empty([0, 2])
        for z in set(r.run(
            renderapi.stack.get_z_values_for_stack, a)).intersection(
                set(r.run(renderapi.stack.get_z_values_for_stack, b))):
            atileIdtoTiles = {ts.tileId: ts for ts in r.run(
                renderapi.tilespec.get_tile_specs_from_z, a, z)}
            btileIdtoTiles = {ts.tileId: ts for ts in r.run(
                renderapi.tilespec.get_tile_specs_from_z, b, z)}
            # only interested in tiles in both stacks
            tilestomatch = (atileIdtoTiles.viewkeys() &
                            btileIdtoTiles.viewkeys())
            self.logger.debug(
                'matching {} tiles from z {}'.format(len(tilestomatch), z))

            # generate centerpoints to use render for dest pts
            #     TODO is it worthwhile to generate a grid/mesh as in PEA?
            centerpoint_l2win = [
                {'tileId': tileId, 'visible': False,
                 'local': [atile.width // 2, atile.height // 2]}
                for tileId, atile in atileIdtoTiles.iteritems()
                if tileId in tilestomatch]

            # get world coordinates for a and b
            wc_a = renderapi.coordinate.local_to_world_coordinates_batch(
                a, centerpoint_l2win, z,
                number_of_threads=self.args['pool_size'], render=r)
            wc_b = renderapi.coordinate.local_to_world_coordinates_batch(
                b, centerpoint_l2win, z,
                number_of_threads=self.args['pool_size'], render=r)

            # format world coordinate json to matching numpy arrays
            acoord = numpy.vstack([
                acoord, numpy.array([d['world'][:2] for d in wc_a])])
            bcoord = numpy.vstack([
                bcoord, numpy.array([d['world'][:2] for d in wc_b])])

        # initialize homography tform and then estimate
        tform = self.__class__.transform_classes[
            self.args['transform_type']]()
        tform.estimate(bcoord, acoord, return_params=False)
        self.logger.info('transform found: {}'.format(tform))

        out_json = {'transform': tform}
        with open(self.args['output_json'], 'w') as f:
            renderapi.utils.renderdump(out_json, f, indent=2)


class RegisterSubvolumeParameters(RenderParameters):
    stack_a = mm.fields.Str(required=True, metadata={
        'description': '"parent" stack to reman fixed'})
    stack_b = mm.fields.Str(required=True, metadata={
        'description': '"child" stack which moves to register to stack_a'})
    transform_type = OptionList(
        RegisterSubvolumeModule.transform_classes.keys(),
        required=False, default='RIGID', metadata={
            'description': 'Homography model to fit'})
    pool_size = mm.fields.Int(required=False, default=1, metadata={
        'description': 'multiprocessing pool size'})


if __name__ == "__main__":
    mod = RegisterSubvolumeModule(
        schema_type=RegisterSubvolumeParameters, input_data=example_parameters)
    mod.run()
