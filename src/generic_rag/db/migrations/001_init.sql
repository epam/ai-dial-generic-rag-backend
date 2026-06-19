create table documents
(
    channel_key  varchar(256)  not null,
    document_id  bigint        not null,
    url          varchar(2048) not null,
    display_name varchar(2048) not null,
    mime_type    varchar(128)  not null,
    size         bigint        not null default 0,
    metadata     jsonb         not null default '{}'::json,
    created_at   timestamp     not null default now(),
    updated_at   timestamp     not null default now(),

    primary key (channel_key, document_id)
);

create unique index on documents (url);

create table text_chunks
(
    channel_key varchar(256) not null,
    document_id bigint       not null,
    chunk_id    bigint       not null,
    page_number integer,
    text        text         not null,

    primary key (channel_key, document_id, chunk_id),
    foreign key (channel_key, document_id)
        references documents (channel_key, document_id) on delete cascade
);

create table image_chunks
(
    channel_key varchar(256)  not null,
    document_id bigint        not null,
    chunk_id    bigint        not null,
    page_number integer,
    image_type  varchar(15)   not null,
    image_url   varchar(2048) not null,
    mime_type   varchar(128)  not null,

    primary key (channel_key, document_id, chunk_id),
    foreign key (channel_key, document_id)
        references documents (channel_key, document_id) on delete cascade
);
