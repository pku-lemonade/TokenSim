"""Audited eager 1A1F API ordering, independent of remote tensor readiness."""
from dataclasses import replace
import simpy


class NativeOrderMixin:
    def _native_control_message(self, *args, **kwargs):
        try:
            yield from self._run_planned_communication(*args, **kwargs)
        except simpy.Interrupt:
            return
        except Exception as exc:
            return exc

    def _start_native_call_order(self, block, state):
        state['calls']={}
        state['length_cpu_ready']=self.env.event()
        state['a_device_ready']=self.env.event()
        state['a_node']=f'i{block.iteration_id}:control:payload:host_submit'
        length,payload=self._control_nodes(block)
        entries=[(length,-1,()),(payload,-1,('length_cpu_ready',))]
        previous_return=None
        link=self._links[block.link_name]
        for layer in range(self.total_layers):
            nodes=self.plan_builder.build_layer(block,layer,link).nodes
            outbound=next(n for n in nodes if n.kind=='communication' and n.direction=='a_to_f')
            inbound=next(n for n in nodes if n.kind=='communication' and n.direction=='f_to_a')
            entries.extend([(outbound,layer,(previous_return,) if previous_return else ()),
                            (inbound,layer,(outbound.node_id,))])
            previous_return=inbound.node_id
        for node,_,_ in entries:
            state['calls'][node.node_id]={'submitted':self.env.event()}
        for node,layer,dependencies in entries:
            state['calls'][node.node_id]['receiver']=self.env.process(
                self._native_receive_call(block,layer,node,state,dependencies))

    def _native_receive_call(self, block, layer, node, state, dependencies):
        try:
            predecessors=[]
            for dependency in dependencies:
                if dependency=='length_cpu_ready':
                    yield state[dependency]
                    predecessors.append(f'i{block.iteration_id}:control:length_readback')
                else:
                    yield state['calls'][dependency]['submitted']
                    predecessors.append(dependency+':host_submit')
            field=self._native_call_cost_field(node,'receive')
            yield from self._run_receiver_submit(block,layer,
                replace(node,predecessor_ids=tuple(predecessors)),node.node_id+':receive_submit',
                self._host_cost_us(field,block)/1e6,field)
        except simpy.Interrupt:
            return
        except Exception as exc:
            # The communication owner rethrows this under its normal DAG guard.
            return exc

    def _run_native_order_control(self, block, state):
        timing=self.cluster_config.af.communication_timing
        length,payload=self._control_nodes(block)
        a,f=length.physical_workers
        prepared=yield from self._wait_step_preparation(block,(a,))
        encoded=yield from self._control_stage(block,length,'encode',prepared,a,
            self._host_cost_us('control_encode_us',block))
        staged=yield from self._control_stage(block,length,'h2d',(encoded,),a,
            timing.control_h2d_us,device=True)
        link=self._links[block.link_name]
        children=[]
        try:
            length_process=self.env.process(self._native_control_message(block,-1,link,
                replace(length,predecessor_ids=(staged,))))
            children.append(length_process)
            yield self.env.any_of((state['calls'][length.node_id]['submitted'],length_process))
            if length_process.triggered and isinstance(length_process.value,Exception):
                raise length_process.value
            payload_process=self.env.process(self._native_control_message(block,-1,link,
                replace(payload,predecessor_ids=(length.node_id+':host_submit',)),
                transport_completed=state['a_device_ready']))
            children.append(payload_process)
            result=yield length_process
            if isinstance(result,Exception):raise result
            yield from self._control_stage(block,length,'length_readback',(length.node_id,),f,
                timing.control_length_readback_us,device=True)
            state['length_cpu_ready'].succeed()
            result=yield payload_process
            if isinstance(result,Exception):raise result
            copied=yield from self._control_stage(block,payload,'payload_readback',(payload.node_id,),f,
                timing.control_payload_readback_us,device=True)
            decoded=yield from self._control_stage(block,payload,'decode',(copied,),f,
                self._host_cost_us('control_decode_us',block))
            if self.operation_profile is not None and 'f_dp_metadata' in self.operation_profile.document['operations']:
                state['f_node']=yield from self._control_stage(block,payload,'f_dp_metadata',(decoded,),f,0,device=True)
            state['f_ready'].succeed()
        finally:
            for child in children:
                if child.is_alive:child.interrupt('native control cancelled')
            if children:
                yield self.env.all_of(children)
