import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

st.header("Flowchart", divider="red")

nodes = [StreamlitFlowNode('1', (200, 0), {'content': 'User Selection'}, 'default', 'bottom', draggable=False, style={'fontSize': '13px', 'color': 'black', 'backgroundColor': '#eaeded', 'border': '2px solid black'}),
		StreamlitFlowNode('2', (50, 100), {'content': '책 정보조회'}, 'default', 'bottom', 'top', draggable=False, style={'fontSize': '13px', 'color': 'black', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),
		StreamlitFlowNode('3', (350, 100), {'content': '책 판매등급'}, 'default', 'bottom', 'top', draggable=False, style={'fontSize': '13px', 'color': 'black', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),

        StreamlitFlowNode('4', (50, 200), {'content': 'ISBN 제공'}, 'default', 'bottom', 'top', draggable=False, style={'fontSize': '13px', 'width': '90px', 'color': 'black', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),
        StreamlitFlowNode('5', (350, 200), {'content': '이미지 제공'}, 'default', 'bottom', 'top', draggable=False, style={'fontSize': '13px', 'width': '90px', 'color': 'black', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),

        StreamlitFlowNode('6', (0, 300), {'content': '알라딘 API 검색'}, 'default', 'bottom', 'top', draggable=False, style={'fontSize': '12px', 'color': 'black','width': '90px', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),
        StreamlitFlowNode('7', (100, 300), {'content': 'ISBN을 제공해주세요'}, 'default', 'top', draggable=False, style={'fontSize': '12px', 'color': 'black', 'width': '90px','backgroundColor': '#ecf0f1', 'border': '2px solid black'}),
        StreamlitFlowNode('8', (300, 300), {'content': 'Load detection model'}, 'default', 'bottom', 'top', draggable=False, style={'fontSize': '12px', 'color': 'black','width': '90px', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),
        StreamlitFlowNode('9', (400, 300), {'content': '사진을 제공해 주세요 '}, 'default', 'top', draggable=False, style={'fontSize': '12px', 'color': 'black','width': '90px', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),

        StreamlitFlowNode('10', (0, 400), {'content': 'Generate description using Gemini API'}, 'default', 'top', draggable=False, style={'fontSize': '10px','width': '90px', 'color': 'black', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),
        StreamlitFlowNode('11', (300, 400), {'content': 'Generate description using Gemini API'}, 'default', 'top', draggable=False, style={'fontSize': '10px','width': '90px', 'color': 'black', 'backgroundColor': '#ecf0f1', 'border': '2px solid black'}),
        
        #StreamlitFlowNode('11', (400, 300), {'content': 'Please provide book images'}, 'default', 'right', 'top', draggable=False, style={'fontSize': '10px', 'padding': 0, 'width': '60px'}),



        ]

edges = [StreamlitFlowEdge('1-2', '1', '2', animated=True, marker_end={'type': 'arrow'}),
		StreamlitFlowEdge('1-3', '1', '3', animated=True, marker_end={'type': 'arrow'}),

        StreamlitFlowEdge('2-4', '2', '4', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('3-5', '3', '5', animated=True, marker_end={'type': 'arrow'}),

        StreamlitFlowEdge('4-6', '4', '6', animated=True, label="YES", label_show_bg=True, label_bg_style={'stroke': 'red', 'fill': 'gray'}, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('4-7', '4', '7', animated=True, label="NO", label_show_bg=True, label_bg_style={'stroke': 'red', 'fill': 'gray'}, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('5-8', '5', '8', animated=True, label="YES", label_show_bg=True, label_bg_style={'stroke': 'red', 'fill': 'gray'}, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('5-9', '5', '9', animated=True, label="NO", label_show_bg=True, label_bg_style={'stroke': 'red', 'fill': 'gray'}, marker_end={'type': 'arrow'}),
        
        StreamlitFlowEdge('6-10', '6', '10', animated=True),
        StreamlitFlowEdge('8-11', '8', '11', animated=True),
        # StreamlitFlowEdge('7-4', '7', '4', animated=True),
        # StreamlitFlowEdge('9-5', '9', '5', animated=True),
        ]

state = StreamlitFlowState(nodes, edges)

streamlit_flow('custom_style_flow',
				state,
				fit_view=True,
				show_minimap=False,
				show_controls=False,
				pan_on_drag=True,
				allow_zoom=True)
